import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 



import re
# Column names list



column_list= ['Column'+str(i) for i in range(15)]

column_list.insert(0, 'Review')

column_list
# Reading the data

data= pd.read_csv('../input/sentisum/sentisum-evaluation-dataset.csv', names= column_list)
# Creating Column16 that would contain all the sentiments joined with ',' for each review.



data['Column16'] = data[data.columns[1:]].apply(lambda x: ', '.join(x.dropna().astype(str)),axis=1)
# Dataset contains redacted company or application names therefore replaced it with 'the company' and created a new column 'Review0'.



data['Review0']= data.apply(lambda x: x['Review'].replace('[REDACTED]', 'the company'), axis=1)
data['Column16'].isnull().sum()



#Inference: No null row in Column16 (Hurrayyy!!)
# Creating a feature to check the presence of [REDACTED] in review.



data['is_redacted']= data.apply(lambda x: '[REDACTED]' in x['Review'], axis=1)
data['is_redacted'].value_counts()



#Inference: 1631 rows have redacted names
data.head()
data.columns
# Creating a review DataFrame for gaining an insight on total type of reviews present in the original dataset



review= pd.DataFrame(data['Column0'].value_counts())



for i in data.columns[2:16]:

    review= pd.concat([review, pd.DataFrame(data[i].value_counts())])
review.shape



#Inference: 282 type of sentiments are present in the entire dataset.
# 'Sum' column contains the count of each sentiment with their polarity tags 



review['Sum']= review.fillna(0).apply(lambda x: sum(x), axis=1)

review.reset_index(inplace= True)

review.rename(columns= {'index':'Reviews'}, inplace= True)
review.head()
pos= review[review['Reviews'].str.contains("positive")]

neg= review[review['Reviews'].str.contains("negative")]

print('Total no. of sentiments with positive polarity tag : {}'.format(pos['Sum'].sum()))

print('Total no. of sentiments with negative polarity tag : {}'.format(neg['Sum'].sum()))

print('Total size of data: {}'.format(data.shape[0]))





# Inference: Clearly there is class imbalance. We need to be more careful regarding negative sentiments since they are very less. 

# Also, there are many rows with overlapping reviews i.e. they contain both negative and positive.

y= [pos['Sum'].sum(), neg['Sum'].sum()]

x= ['Total no. of sentiments with positive polarity tag', 'Total no. of sentiments with negative polarity tag']



plt.figure(figsize = (10,5))

plt.barh(x,y, color = "m")

plt.grid(True)
labels= list(review.iloc[:10]['Reviews'])

size= list(review.iloc[:10]['Sum'])

explode= [0.1, 0.1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.3]



plt.figure(figsize = (9,9))

plt.pie(size, labels = labels, shadow = True, startangle = 90, autopct='%1.1f%%', explode= explode )

plt.title("Reviews Distribution", y=1.095)

plt.legend()



#Inference: The data is biased towards positive polarity and 'value for money' is the most used sentiment in terms of frequency.
x= data[data['Column16'].str.contains('positive')].shape[0]

y= data[data['Column16'].str.contains('negative')].shape[0]

z= data[~data['Column16'].str.contains('positive')].shape[0]



print('Total rows containing at least one positive sentiment: ', x)

print('Total rows containing at least one negative sentiment: ', y)

print('Total rows containing neither positive nor negative sentiment: ', z-y)
ax= [x,y,z]

ordinate= ['Total rows containing at least one positive sentiment', 'Total rows containing at least one negative sentiment','Total rows containing neither positive nor negative sentiment']



plt.figure(figsize = (10,5))

plt.barh(ordinate,ax, color = "b")

plt.grid(True)
# For eg. this cell contains no explicit positive or negative tag



data['Column16'][10130]
# Dropping the rows where polarity tag is not provided since it would make the trained model less impactful

 

data.drop(data[~data['Column16'].str.contains('negative|positive')].index, inplace= True)
data.tail(3)  # Indexing is wrong, reset is required
data.reset_index(inplace=True)

data= data.drop(['index'],axis=1)
data= data.drop([])
data.head(3)
# Sterilization function for cleaning the data and making it more suitable for the model.



def sterilization(data):

    

    data = re.sub('https?://\S+|www\.\S+', '', data) #remove any HTTP link

    data = re.sub('<*>', '', data)

    emoj = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    emoj.sub(r'', data)

    data = re.sub(r'\w*\d\w*','', data) #remove any number of alphanumeric shortcuts used in comments

    data = re.sub(r'\s+', ' ', data)    #remove white space character



    return data
# Cleaning data...

data['Review0']=data['Review0'].apply(lambda x : sterilization(x))
# Changing column order for better view of dataset



data= data.drop(['Column0', 'Column1', 'Column2', 'Column3', 'Column4',

       'Column5', 'Column6', 'Column7', 'Column8', 'Column9', 'Column10',

       'Column11', 'Column12', 'Column13', 'Column14' ], axis=1)

data.tail()
import glob

import json

import time

import logging

import random

import re

from itertools import chain

from string import punctuation





import pandas as pd

import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader
!pip install pytorch_lightning==0.8.1
import argparse

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
# Dividing dataset for train, validation and testing



train= data[:7001]

val= data[7001:8101]

ref= data[8101:]

train.to_csv('train.csv', index=False)

val.to_csv('valid.csv', index= False)

ref.to_csv('ref.csv', index=False)
class T5FineTuner(pl.LightningModule):

    def __init__(self, hparams):

        super(T5FineTuner, self).__init__()

        self.hparams = hparams



        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)



    def is_logger(self):

        return True



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

        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,

                                num_workers=4)

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

        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="valid", args=self.hparams)

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

    max_seq_length=200,

    learning_rate=3e-4,

    weight_decay=0.0,

    adam_epsilon=1e-8,

    warmup_steps=0,

    train_batch_size=16,

    eval_batch_size=16,

    num_train_epochs=2,

    gradient_accumulation_steps=8,

    n_gpu=1,

    early_stop_callback=False,

    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true

    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties

    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default

    seed=42,

)
class SentimentDataset(Dataset):

    def __init__(self, tokenizer, data_dir, type_path, max_len=200):

        self.path = os.path.join(data_dir, type_path + '.csv')



        self.question = "Review0"             # Review0 column as input

        self.target_column = "Column16"       # Column16 column as output

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

            target,question= self.data.loc[idx, self.target_column], self.data.loc[idx, self.question]



            input_ = "Comment: %s </s>" % (question)

            target = " %s </s>" %(target)



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
# Pretrained tokenizer is used



tokenizer = T5Tokenizer.from_pretrained('t5-base')
# Gentle check if everything is according to the plan or not



dataset = SentimentDataset(tokenizer, '/kaggle/working', 'valid', 200)



data = dataset[200]

print(tokenizer.decode(data['source_ids']))

print(tokenizer.decode(data['target_ids']))
!mkdir result
# We will train the model for 6 epochs. This value is decided through iterative process.



args_dict.update({'data_dir': '/kaggle/working', 'output_dir': '/kaggle/working/result', 'num_train_epochs':1,'max_seq_length':200})

args = argparse.Namespace(**args_dict)

print(args_dict)
checkpoint_callback = pl.callbacks.ModelCheckpoint(

    

    period =1,filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1

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

    return SentimentDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)
print ("Initialize model")

model = T5FineTuner(args)



trainer = pl.Trainer(**train_params)
print (" Training model")

trainer.fit(model)



print ("training finished")



print ("Saving model")

model.model.save_pretrained("/kaggle/working/result")



print ("Saved model")