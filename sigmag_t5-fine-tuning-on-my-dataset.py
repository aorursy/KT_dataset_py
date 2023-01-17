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
!nvidia-smi
!pip install transformers
!pip install pytorch_lightning==0.9.0
# Import libraries
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
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=None):
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
    train_batch_size=2,
    eval_batch_size=2,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
import csv
from dataclasses import dataclass

from enum import Enum
from typing import List, Optional
from transformers import PreTrainedTokenizer
def generate_real_sample(row):
  assert("name" in row)
  assert("occupation" in row)

  name = row["name"]
  occupation = row["occupation"]

  output = f"{name} is a reputed {occupation}"

  if "born" in row:
    born = row["born"]
    output += f" who was born in {born}."
  else:
    output += "."
  
  return output

!curl -LO https://gist.github.com/wsc/1083459/raw/d8d0aa8737a36912e6c119a172c8367276b76260/gistfile1.txt
with open("gistfile1.txt") as f:
  occupations = f.read().split("\n")
print(occupations[:5])

ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
achievement = ["youngest", "tallest", "smallest", "oldest", ]

def generate_fake_sample(row):
  assert("name" in row)
  assert("occupation" in row)

  fake_ac = random.choice(achievement)
  fake_or = random.choice(ordinals)

  or_text = fake_ac if fake_or == 'first' else fake_or + " " + fake_ac

  name = row["name"]
  occupation = row["occupation"]
  output = f"{name} is a reputed {occupation} and he was the {or_text} to do so."

  return output

def get_random_proper_noun():
  le = np.random.randint(4, 10)
  st = [chr(np.random.randint(97, 97 + 26)) for _ in range(le)]
  st[0] = chr(ord(st[0]) - 32)
  st = "".join(st)
  return st

def get_random_occupation():
  return random.choice(occupations)

def get_datarow():
  data = {
      "name": get_random_proper_noun(), # TODO: using real proper nouns
      "occupation": get_random_occupation()
  }

  if np.random.random() < 0.5:
    data["born"] = get_random_proper_noun()

  return data

def to_str(data):
  out = ""
  
  first = True

  for key, value in data.items():
    out += ("" if first else " ") + f"{key} IS {value}."
    first = False

  return out
REAL_COUNT = 5000
FAKE_COUNT = 500

real_samples = []
fake_samples = []

for _ in range(REAL_COUNT):
  data = get_datarow()
  data_str = to_str(data)
  sample = generate_real_sample(data)
  
  real_samples.append((data_str, sample))

for _ in range(FAKE_COUNT):
  data = get_datarow()
  data_str = to_str(data)
  sample = generate_fake_sample(data)

  fake_samples.append((data_str, sample))
    
dataset = real_samples + fake_samples
df = pd.DataFrame(dataset, columns=["prompt", "response"])
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class InputExample:
  prompt: str
  response: str


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class MyProcessor:
    def __init__(self, df):
        train, test = train_test_split(df)
        train, dev = train_test_split(train)
        self.train = train.values.tolist()
        self.test = test.values.tolist()
        self.dev = dev.values.tolist()

    def get_train_examples(self):        
        return [InputExample(x, y) for x, y in self.train]

    def get_dev_examples(self):
        return [InputExample(x, y) for x, y in self.dev]

    def get_test_examples(self):
        return [InputExample(x, y) for x, y in self.test]
class MyDataset(Dataset):
  def __init__(self, tokenizer, data, type_path,max_len=512):
    self.type_path = type_path
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self.proc = MyProcessor(data)

    self._build()
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def __len__(self):
    return len(self.inputs)
  
  def _build(self):
    if self.type_path == 'train':
      examples = self.proc.get_train_examples()
    else:
      examples = self.proc.get_dev_examples()
    
    for example in examples:
      self._create_features(example)
  
  def _create_features(self, example):
    input_ = "prompt: %s</s>" % (example.prompt)
    target = "%s </s>" % (str(example.response))

    # tokenize inputs
    tokenized_inputs = self.tokenizer.batch_encode_plus(
        [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
    )
    # tokenize targets
    tokenized_targets = self.tokenizer.batch_encode_plus(
        [target], max_length=150, pad_to_max_length=True, return_tensors="pt"
    )

    self.inputs.append(tokenized_inputs)
    self.targets.append(tokenized_targets)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
dataset = MyDataset(tokenizer, df, type_path='val')
len(dataset)
data = dataset[67]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))
!mkdir -p /kaggle/working/t5_text
args_dict.update({ 'output_dir': '/kaggle/working/t5_text/', 'num_train_epochs': 1})
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
  return MyDataset(tokenizer=tokenizer, data=df, type_path=type_path,  max_len=args.max_seq_length)
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
%env JOBLIB_TEMP_FOLDER=/tmp

trainer.fit(model)
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
dataset =  MyDataset(tokenizer, data=df, type_path="val",  max_len=args.max_seq_length)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'], 
                              attention_mask=batch['source_mask'], 
                              max_length=200)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  outputs.extend(dec)
  targets.extend(target)
print(targets[:10])
print("----\n")
print(outputs[:10])