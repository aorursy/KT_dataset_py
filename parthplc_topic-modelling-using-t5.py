# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
news = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json",lines=True)
news.shape
news.columns
news.head()
pd.options.display.max_rows

pd.set_option('display.max_colwidth', -1)
news.headline
news.short_description
news["category"].value_counts()
news.shape
short = news.sample(frac = 0.1,random_state = 1)
short.shape
short["category"].value_counts()
short.columns
!nvidia-smi
!pip install transformers

!pip install pytorch_lightning
import argparse

import glob

import os

import json

import time

import logging

import random

import re

from itertools import chain

from string import punctuation



import nltk

nltk.download('punkt')

from nltk.tokenize import sent_tokenize



import pandas as pd

import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl





from transformers import (

    AdamW,

    T5ForConditionalGeneration,

    T5Tokenizer,

    get_linear_schedule_with_warmup

)
def set_seed(seed):

  random.seed(seed)

  np.random.seed(seed)

  torch.manual_seed(seed)

  if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



set_seed(42)
class T5FineTuner(pl.LightningModule):

  def __init__(self, hparams):

    super(T5FineTuner, self).__init__()

    self.hparams = hparams

    

    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  

  def is_logger(self):

    return self.trainer.proc_rank <= 0

  

  def forward(

      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None

  ):

    return self.model(

        input_ids,

        attention_mask=attention_mask,

        decoder_input_ids=decoder_input_ids,

        decoder_attention_mask=decoder_attention_mask,

        lm_labels=lm_labels,

    )



  def _step(self, batch):

    lm_labels = batch["target_ids"]

    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100



    outputs = self(

        input_ids=batch["source_ids"],

        attention_mask=batch["source_mask"],

        lm_labels=lm_labels,

        decoder_attention_mask=batch['target_mask']

    )



    loss = outputs[0]



    return loss



  def training_step(self, batch, batch_idx):

    loss = self._step(batch)



    tensorboard_logs = {"train_loss": loss}

    return {"loss": loss, "log": tensorboard_logs}

  

  def training_epoch_end(self, outputs):

    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()

    tensorboard_logs = {"avg_train_loss": avg_train_loss}

    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}



  def validation_step(self, batch, batch_idx):

    loss = self._step(batch)

    return {"val_loss": loss}

  

  def validation_epoch_end(self, outputs):

    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

    tensorboard_logs = {"val_loss": avg_loss}

    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}



  def configure_optimizers(self):

    "Prepare optimizer and schedule (linear warmup and decay)"



    model = self.model

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [

        {

            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],

            "weight_decay": self.hparams.weight_decay,

        },

        {

            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],

            "weight_decay": 0.0,

        },

    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

    self.opt = optimizer

    return [optimizer]

  

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):

    if self.trainer.use_tpu:

      xm.optimizer_step(optimizer)

    else:

      optimizer.step()

    optimizer.zero_grad()

    self.lr_scheduler.step()

  

  def get_tqdm_dict(self):

    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}



    return tqdm_dict



  def train_dataloader(self):

    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)

    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)

    t_total = (

        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))

        // self.hparams.gradient_accumulation_steps

        * float(self.hparams.num_train_epochs)

    )

    scheduler = get_linear_schedule_with_warmup(

        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total

    )

    self.lr_scheduler = scheduler

    return dataloader



  def val_dataloader(self):

    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)

    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)
logger = logging.getLogger(__name__)



class LoggingCallback(pl.Callback):

  def on_validation_end(self, trainer, pl_module):

    logger.info("***** Validation results *****")

    if pl_module.is_logger():

      metrics = trainer.callback_metrics

      # Log results

      for key in sorted(metrics):

        if key not in ["log", "progress_bar"]:

          logger.info("{} = {}\n".format(key, str(metrics[key])))



  def on_test_end(self, trainer, pl_module):

    logger.info("***** Test results *****")



    if pl_module.is_logger():

      metrics = trainer.callback_metrics



      # Log and save results to file

      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")

      with open(output_test_results_file, "w") as writer:

        for key in sorted(metrics):

          if key not in ["log", "progress_bar"]:

            logger.info("{} = {}\n".format(key, str(metrics[key])))

            writer.write("{} = {}\n".format(key, str(metrics[key])))
args_dict = dict(

    data_dir="", # path for data files

    output_dir="", # path to save the checkpoints

    model_name_or_path='t5-base',

    tokenizer_name_or_path='t5-base',

    max_seq_length=512,

    learning_rate=3e-4,

    weight_decay=0.0,

    adam_epsilon=1e-8,

    warmup_steps=0,

    train_batch_size=8,

    eval_batch_size=8,

    num_train_epochs=2,

    gradient_accumulation_steps=16,

    n_gpu=1,

    early_stop_callback=False,

    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true

    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties

    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default

    seed=42,

)
train = news.sample(frac = 0.1,random_state = 1)

val = news.sample(frac = 0.01,random_state = 41)
train.shape,val.shape
train.columns
drop_list = ['authors', 'link', 'date']

train.drop(drop_list,axis = 1,inplace = True)

val.drop(drop_list,axis = 1,inplace = True)
!pwd
!mkdir data

train.to_csv("/kaggle/working/data/train.csv",index = False)

val.to_csv("/kaggle/working/data/val.csv",index = False)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
train.columns
class Topicmodel(Dataset):

    def __init__(self, tokenizer, data_dir, type_path, max_len=256):

        self.path = os.path.join(data_dir, type_path + '.csv')



        self.source_column_1 = 'headline'

        self.source_column_2 = 'short_description'

        self.target_column = 'category'

        self.data = pd.read_csv(self.path)

        self.max_len = max_len

        self.tokenizer = tokenizer

        self.inputs = []

        self.targets = []



        self._build()



    def __len__(self):

        return len(self.inputs)



    def __getitem__(self, index):

        source_ids = self.inputs[index]["input_ids"].squeeze()

        target_ids = self.targets[index]["input_ids"].squeeze()



        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze

        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze



        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}



    def _build(self):

        for idx in range(len(self.data)):

            input_1,input_2,target = self.data.loc[idx, self.source_column_1],self.data.loc[idx, self.source_column_2],self.data.loc[idx, self.target_column]



            input_ = "Text : "+ str(input_1) + "."+ str(input_2)  +"</s>"

            target = "Topic : " +str(target) + " </s>"



            # tokenize inputs

            tokenized_inputs = self.tokenizer.batch_encode_plus(

                [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"

            )

            # tokenize targets

            tokenized_targets = self.tokenizer.batch_encode_plus(

                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"

            )



            self.inputs.append(tokenized_inputs)

            self.targets.append(tokenized_targets)
!pwd
dataset = Topicmodel(tokenizer, '/kaggle/working/data/', 'val', 312)

print("Val dataset: ",len(dataset))
data = dataset[61]

print(tokenizer.decode(data['source_ids']))

print(tokenizer.decode(data['target_ids']))
if not os.path.exists('t5_data'):

    os.makedirs('t5_data')
!ls
!ls t5_data
args_dict.update({'data_dir': '/kaggle/working/data/', 'output_dir': 't5_data', 'num_train_epochs':1,'max_seq_length':312})

args = argparse.Namespace(**args_dict)

print(args_dict)
checkpoint_callback = pl.callbacks.ModelCheckpoint(

    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5

)



train_params = dict(

    accumulate_grad_batches=args.gradient_accumulation_steps,

    gpus=args.n_gpu,

    max_epochs=args.num_train_epochs,

    early_stop_callback=False,

    precision= 16 if args.fp_16 else 32,

    amp_level=args.opt_level,

    gradient_clip_val=args.max_grad_norm,

    checkpoint_callback=checkpoint_callback,

    callbacks=[LoggingCallback()],

)

def get_dataset(tokenizer, type_path, args):

  return Topicmodel(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)
print ("Initialize model")

model = T5FineTuner(args)



trainer = pl.Trainer(**train_params)



print (" Training model")

trainer.fit(model)
print ("training finished")



print ("Saving model")

model.model.save_pretrained('t5_data')



print ("Saved model")
import textwrap

from tqdm.auto import tqdm

from sklearn import metrics
!pwd
dataset =  Topicmodel(tokenizer, data_dir='/kaggle/working/data/', type_path='val')

loader = DataLoader(dataset, batch_size=32, num_workers=4)
outputs = []

targets = []

for batch in tqdm(loader):

  outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 

                              attention_mask=batch['source_mask'].cuda(), 

                              max_length=20)



  dec = [tokenizer.decode(ids) for ids in outs]

  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

  

  outputs.extend(dec)

  targets.extend(target)
outputs
metrics.accuracy_score(targets, outputs)
def topic(string):

    text = "Text : " + string + "</s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")

#     input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outs = model.model.generate(input_ids=encoding["input_ids"].cuda(), 

                              attention_mask=encoding["attention_mask"].cuda(), 

                              max_length=20)

    print ("\nOriginal Text ::")

    print (string)

    print ("Topic :: ")

    string_final = [tokenizer.decode(ids) for ids in outs]    

    return(" ".join(string_final))





    
str1 = "sachin tendulkar"

topic(str1)
str1 = "Donald Trump"

topic(str1)
str1 = "pharma"

topic(str1)
# Merge both Title and description

# !wget http://blob.thijs.ai/wiki-summary-dataset/raw.tar.gz
# !ls
# !unzip raw.tar.gz
# !ls
# !tar --gunzip --extract --verbose --file=raw.tar.gz
# df = pd.read_csv('raw.txt', sep="|", header=None, names=['Topic','Nan','Nan2','Text'])
# df.shape
# df.head()
# drop_list = ["Nan","Nan2"]

# df.drop(drop_list,inplace = True,axis = 1 )
# new_df = df.sample(frac=0.01, replace=True, random_state=1)
# new_df.shape
# new_df.tail(10)
# df = pd.read_csv("/kaggle/input/medium-stories/Medium_Clean.csv")
# print(df.columns)
# drop_list = ['Image', 'Author', 'Publication','Year', 'Month', 'Day', 'Reading_Time']

# df.drop(drop_list,axis = 1,inplace = True)
# drop_list = ['Unnamed: 0','Claps', 'url', 'Author_url']









# df.drop(drop_list,axis = 1,inplace = True)
# df.columns
# df.shape
# df.isna().sum()
# df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# df.shape
# listl = ['Tag_ai', 'Tag_android', 'Tag_apple','Tag_architecture', 'Tag_art', 'Tag_artificial-intelligence','Tag_big-data', 'Tag_bitcoin', 'Tag_blacklivesmatter', 'Tag_blockchain',

#        'Tag_blog', 'Tag_blogging', 'Tag_books', 'Tag_branding', 'Tag_business',

#        'Tag_college', 'Tag_computer-science', 'Tag_creativity',

#        'Tag_cryptocurrency', 'Tag_culture', 'Tag_data', 'Tag_data-science',

#        'Tag_data-visualization', 'Tag_deep-learning', 'Tag_design', 'Tag_dogs',

#        'Tag_donald-trump', 'Tag_economics', 'Tag_education', 'Tag_energy',

#        'Tag_entrepreneurship', 'Tag_environment', 'Tag_ethereum',

#        'Tag_feminism', 'Tag_fiction', 'Tag_food', 'Tag_football', 'Tag_google',

#        'Tag_government', 'Tag_happiness', 'Tag_health', 'Tag_history',

#        'Tag_humor', 'Tag_inspiration', 'Tag_investing', 'Tag_ios',

#        'Tag_javascript', 'Tag_jobs', 'Tag_journalism', 'Tag_leadership',

#        'Tag_life', 'Tag_life-lessons', 'Tag_love', 'Tag_machine-learning',

#        'Tag_marketing', 'Tag_medium', 'Tag_mobile', 'Tag_motivation',

#        'Tag_movies', 'Tag_music', 'Tag_nba', 'Tag_news', 'Tag_nutrition',

#        'Tag_parenting', 'Tag_personal-development', 'Tag_photography',

#        'Tag_poem', 'Tag_poetry', 'Tag_politics', 'Tag_product-design',

#        'Tag_productivity', 'Tag_programming', 'Tag_psychology', 'Tag_python',

#        'Tag_racism', 'Tag_react', 'Tag_relationships', 'Tag_science',

#        'Tag_self-improvement', 'Tag_social-media', 'Tag_software-engineering',

#        'Tag_sports', 'Tag_startup', 'Tag_tech', 'Tag_technology', 'Tag_travel',

#        'Tag_trump', 'Tag_ux', 'Tag_venture-capital', 'Tag_web-design',

#        'Tag_web-development', 'Tag_women', 'Tag_wordpress', 'Tag_work',

#        'Tag_writing']
# listl[0]
# len(df)









# df["Tags"] = " "
# df.reset_index(inplace = True)
# def tag(df):

#     for i in range(10000):

#         for j in listl:

#             if df[j][i] == int(1):

#                 df["Tags"][i] = str(df["Tags"][i]) + " " +str(j)

# %%timeit

# tag(df)
# df["Tags"]