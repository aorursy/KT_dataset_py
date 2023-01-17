VERSION = "nightly"  
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
!git clone https://github.com/huggingface/transformers.git
!pip install ./transformers

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('t5-base')
# process the examples in input and target text format and the eos token at the end 
def add_eos_to_examples(example):
    example['input_text'] = '%s </s>' % (example['question'])
    example['target_text'] = '%s </s>' % (example['answers'])
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings
import pandas as pd
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test.head()
import matplotlib.pyplot as plt
plt.style.use('ggplot')
train['sentiment'].hist()
content = []
for i in range(len(train)):
  content.append("context: %s classification: %s </s>"%(train['text'][i], train['sentiment'][i]))
train['content'] = content
content = []
for i in range(len(test)):
  content.append("context: %s classification: %s </s>"%(test['text'][i], test['sentiment'][i]))
test['content'] = content
def pre(t):
  return "%s </s>"%t
train['selected_text'] = train['selected_text'].apply(pre)
from sklearn.model_selection import train_test_split
train = train[['content','selected_text']]
train.columns = ['text', 'target']
train, valid = train_test_split(train, test_size=0.2, random_state=42)
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

class T5Model(Dataset):
  def __init__(self, tokenizer,df,  max_len=128, train=True):
    self.data_column = df["text"].values
    if train:
        self.class_column = df['target'].values
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.train = train
        
  def __len__(self):
      return len(self.data_column)

  def __getitem__(self, index):
    # tokenize inputs
    tokenized_inputs = self.tokenizer.encode_plus( self.data_column[index], max_length=self.max_len, pad_to_max_length=True, truncation=True,return_tensors="pt")
    source_ids = tokenized_inputs["input_ids"].squeeze()
    src_mask    = tokenized_inputs["attention_mask"].squeeze() # might need to squeeze
    
    if self.train:
        tokenized_targets = self.tokenizer.encode_plus( self.class_column[index] , max_length=32, pad_to_max_length=True,truncation=True, return_tensors="pt")
        target_ids = tokenized_targets["input_ids"].squeeze()
        target_mask = tokenized_targets['attention_mask'].squeeze()  # might need to squeeze
        return {"input_ids": source_ids, "attention_mask": src_mask, 
                "target_ids": target_ids, "target_attention_mask": target_mask}
    else:
        return {"input_ids": source_ids, "attention_mask": src_mask}
train_dataset = T5Model(tokenizer, train)
valid_dataset = T5Model(tokenizer,valid)
test_dataset = T5Model(tokenizer, test, train=False)
%%timeit
train_dataset[1]
len(train_dataset), len(valid_dataset), len(test_dataset)
# cach the dataset, so we can load it directly for training

torch.save(train_dataset, 'train_data.pt')
torch.save(valid_dataset, 'valid_data.pt')
torch.save(test_dataset, 'test_data.pt')
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator():
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'lm_labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file, 
    #make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    train_dataset  = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print('loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
import json
args_dict = {
  "num_cores": 8,
  "model_name_or_path": 't5-base',
  "max_len": 128 ,
  "target_max_len": 2,
  "output_dir": './models/tpu',
  "overwrite_output_dir": True,
  "per_device_train_batch_size": 4,
  "per_gpu_eval_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 5e-5,
  "tpu_num_cores": 8,
  "num_train_epochs": 5,
  "do_train": True
}
with open('args.json', 'w') as f:
  json.dump(args_dict, f)
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
import torch
import torch_xla
import torch_xla.core.xla_model as xm

from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm
model = T5ForConditionalGeneration.from_pretrained('models/tpu').to('cpu') # because its loaded on xla by default
tokenizer = T5Tokenizer.from_pretrained('models/tpu')
valid_dataset = torch.load('valid_data.pt')
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
answers = []
for batch in tqdm(dataloader):
  outs = model.generate(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        max_length=2,
                        early_stopping=True)
  outs = [tokenizer.decode(ids) for ids in outs]
  answers.extend(outs)
predictions = []
references = []
for ref, pred in zip(valid_dataset, answers):
  predictions.append(pred)
  references.append(tokenizer.decode(ref['target_ids']))
predictions[0], references[0]
from sklearn.metrics import classification_report
print(classification_report(references, predictions))
