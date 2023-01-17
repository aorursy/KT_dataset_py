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
!pip install --upgrade pytorch_lightning
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from sklearn.model_selection import train_test_split
import shutil
from string import punctuation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
print("is GPU available: ", torch.cuda.is_available())
print(pl.__version__)
with open('../input/news-data/news_with_ner_predictions.json', encoding='utf-8') as f:
    news_data = json.load(f)

news_items_data = []
doc_cnt = 0
neg_doc_cnt = 0
# Interested only in Organizations & Persons
valid_ner_tags = ['I-ORG', 'B-ORG', 'I-PER', 'B-PER']

for item in news_data:
    temp_item = {}
    temp_item["text"] = item["text"]
    temp_item["labels"] = item["labels"]
    temp_item["token_size"] = item["token_size"]
    temp_item["predicted_ner"] = []
    
    for ner_item in item['predicted_ner']:
        if (ner_item[1] in valid_ner_tags) and \
        (ner_item[0].startswith("##") == False) and \
        (len(ner_item[0]) > 1):
            
            dup_flag = 0
            for val, _ in temp_item["predicted_ner"]:
                if val.strip() == ner_item[0].strip():
                    dup_flag = 1
                    break
    
            if dup_flag == 0:
                temp_item["predicted_ner"].append(ner_item)
            
    if len(temp_item["predicted_ner"]) > 0:
        doc_cnt += 1
        if len(temp_item["labels"]) > 0:
            neg_doc_cnt += 1
        news_items_data.append(temp_item)
        
print("Total paragraphs which has valid NER tags: ", doc_cnt)
print("Total paragraphs which has negative labels: ", neg_doc_cnt)

# Setting random seed and undersample the unrelated class
np.random.seed(99)
final_news_data = []
for item in news_items_data:
    if len(item["labels"]) > 0 or (np.random.rand() > 0.72):
        final_news_data.append(item)
print("Number of samples used for modelling: ", len(final_news_data))
final_news = pd.DataFrame(final_news_data)
final_news.head()
final_news["label_type"] = final_news["labels"].map(lambda x: len(x) > 0).astype(int)
neg_idx_list = final_news[final_news["label_type"] == 1].index.tolist()
pos_idx_list = final_news[final_news["label_type"] == 0].index.tolist()

np.random.seed(99)
np.random.shuffle(neg_idx_list)
np.random.shuffle(pos_idx_list)

val_idx_list = neg_idx_list[-1 * round(len(neg_idx_list) * 0.04):] + \
            pos_idx_list[-1 * round(len(pos_idx_list) * 0.05):]
final_news["split"] = ['val' if i in val_idx_list else 'train' for i in range(final_news.shape[0])]

#!ls -l ../input
#!rmdir final_news_data
!mkdir final_news_data
#final_news[final_news["split"] == 'train'].to_csv("./final_news_data/train.json", index=False)
#final_news[final_news["split"] == 'val'].to_csv("./final_news_data/val.json", index=False)
final_news[final_news["split"] == 'train'].to_json("./final_news_data/train.json", orient='records')
final_news[final_news["split"] == 'val'].to_json("./final_news_data/val.json", orient='records')
class T5FineTuner(pl.LightningModule):
    
    def __init__(self, hparams):
        
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
  
    def is_logger(self):
        return self.trainer.proc_rank <= 0
  
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, \
                  decoder_attention_mask=None, lm_labels=None):
        
        return self.model(input_ids,attention_mask=attention_mask, \
                          decoder_input_ids=decoder_input_ids, \
                          decoder_attention_mask=decoder_attention_mask, \
                          lm_labels=lm_labels)
    
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"], \
                       lm_labels=lm_labels, decoder_attention_mask=batch['target_mask'])
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
            }]
        
        optimizer = AdamW(optimizer_grouped_parameters, \
                          lr=self.hparams.learning_rate, \
                          eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]
    
    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, \
                       second_order_closure=None, on_tpu=False, using_native_amp=False, \
                       using_lbfgs=False):
        
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()
        
  
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, \
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs))
        
        print("Total training steps: ", t_total)
        
        scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
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
    seed=100)
class SwagDataset(Dataset):
    
    def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
        
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.examples = pd.read_json(os.path.join(self.data_dir, self.type_path + ".json"))
  
    def __getitem__(self, index):
        
        
        inputs, targets = self._create_features(self.examples.iloc[index])
        source_ids = inputs["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = targets["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, \
                "target_ids": target_ids, "target_mask": target_mask}
  
    def __len__(self):
        return len(self.examples)
  
    def _build(self):
        #examples = pd.read_csv(os.path.join(self.data_dir, self.type_path + ".csv"))
        #for idx, row in examples.iterrows():
        #    self._create_features(row)
        pass
  
    def _create_features(self, p_row):
        
        src_text = p_row["text"]
        src_content = "context: " + src_text
        rnd_ner = p_row["predicted_ner"][np.random.randint(0, len(p_row["predicted_ner"]))][0]

        if len(p_row["labels"]) > 0:

            match_list = [rnd_ner for label in p_row["labels"] \
                          if label["mapped_para_text"].strip().lower() == rnd_ner.strip().lower()]

            if len(match_list) > 0:
                # Create negative label
                src_content = src_content + " " + "The review on entity " + rnd_ner.strip() + \
                                " is options: 1: negative. 2: non-negative. </s>"
                trg = "1 </s>"
            else:
                # Create non-negative label
                src_content = src_content + " " + "The review on entity " + rnd_ner.strip() + \
                                " is options: 1: negative. 2: non-negative. </s>"
                trg = "2 </s>"
        else:
            # Create unrelated label
            src_content = src_content + " " + "The review on entity " + rnd_ner.strip() + \
                            " is options: 1: negative. 2: non-negative. </s>"
            trg = "2 </s>"
        

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus([src_content], max_length=self.max_len, \
                                                            pad_to_max_length=True, return_tensors="pt")
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus([trg], max_length=2, \
                                                             pad_to_max_length=True, return_tensors="pt")
        
        return tokenized_inputs, tokenized_targets
!mkdir -p t5_swag
data_dir = 'final_news_data'
out_dir = 't5_swag'
num_epochs = 10

tokenizer = T5Tokenizer.from_pretrained('t5-base')
dataset = SwagDataset(tokenizer, data_dir=data_dir, type_path='val')
print("len of val dataset: ", len(dataset))

''' update args '''
args_dict.update({'data_dir': data_dir, 'output_dir': out_dir, 'num_train_epochs':num_epochs})
args = argparse.Namespace(**args_dict)
args.max_grad_norm
#next(iter(dataset))
#pd.read_("./final_news_data/val.csv")["labels"].iloc[6]
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir, prefix="checkpoint", \
                                                   monitor="val_loss", mode="min", save_top_k=5)

''' checkpoint_callback should be given as below argument to persist the model'''
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=None,
    callbacks=[LoggingCallback()])
def get_dataset(tokenizer, type_path, args):
    return SwagDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  \
                       max_len=args.max_seq_length)
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)
!ls -l "/kaggle/working/"
#!rm -r "/kaggle/working/lightning_logs"
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
#torch.cuda.device_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset =  SwagDataset(tokenizer, data_dir=data_dir, type_path='val')
loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)

#dataset = ImdbDataset(tokenizer, 'aclImdb', 'test',  max_len=512)
#loader = DataLoader(dataset, batch_size=32, num_workers=4)

''' Put on GPU, not required if we are checkpointing'''
model.to(device)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['source_ids'].to(device), \
                                attention_mask=batch['source_mask'].to(device), \
                                max_length=2)

    dec = [tokenizer.decode(ids) for ids in outs]
    target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
    outputs.extend(dec)
    targets.extend(target)
#loader = DataLoader(dataset, batch_size=32, shuffle=True)
metrics.accuracy_score(targets, outputs)
print(metrics.classification_report(targets, outputs))
''' Test the results'''
it = iter(loader)
batch = next(it)
batch["source_ids"].shape
outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                              attention_mask=batch['source_mask'].to(device), 
                              max_length=2)

dec = [tokenizer.decode(ids) for ids in outs]

texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
for i in range(batch["source_ids"].shape[0]):
    c = texts[i]
    lines = textwrap.wrap("text:\n%s\n" % c, width=100)
    print("\n".join(lines))
    print("\nActual sentiment: %s" % targets[i])
    print("predicted sentiment: %s" % dec[i])
    print("=====================================================================\n")
