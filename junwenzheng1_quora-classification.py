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
!ls /kaggle/input
df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
df2 = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

df2.head()
df.shape
import transformers
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
from collections import Counter
import nltk
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputFeatures, DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass, field
import tqdm

class Split(Enum):
    train = "train"
    dev = "dev"
@dataclass
class CsvClassifierDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        default='/kaggle/input/quora-insincere-questions-classification/',
        metadata={"help": "The input data dir. Should contain the .csv files (or other data files) for the task."}
    )
    cache_dir: str = field(
        default='/kaggle/temp/',
        metadata={"help": "The cache data dir. Can be used to write the .lock files"}
    )
    source_key: str = field(
        default='question_text',
        metadata={
            "help": "The source key in the csv."
        }
    )
    source_key_b: Optional[str] = field(
        default=None,
        metadata={
            "help": "The source key b in the csv."
        }
    )
    target_key: str = field(
        default='target',
        metadata={
            "help": "The target key in the csv."
        }
    )

    max_seq_length: int = field(
        default=100,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    num_labels: int = field(
        default=2,
        metadata={
            "help": "The number of classification labels."
        },
    )

class CsvClassifierProcessor(DataProcessor):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir, train_file, dev_file, source_key, label_key, source_key_b=None):
        self.label_key = label_key
        self.source_key = source_key
        self.dev_file = dev_file
        self.train_file = train_file
        self.data_dir = data_dir
        self.source_key_b = source_key_b
        self.map = {}
        
    def _get_examples_from_file(self, filepath: Path, split: str):
        print(f'Opening {filepath}')
        examples = []
        data = pd.read_csv(filepath)
        for i, row in tqdm.tqdm(data.iterrows()):
            if type(row[self.source_key]) == str:
                text_b = row.get(self.source_key_b, None)
                if text_b is not None:
                    text_b = text_b.strip()
                label_ = row.get(self.label_key, None)
                ie = InputExample(guid=f"{split}-{i}", text_a=row[self.source_key], text_b=text_b,
                                      label=label_)
                examples.append(ie)
        self.map[filepath] = examples
        return examples

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        filepath = Path(self.data_dir) / self.train_file
        print('Getting train examples')
        if filepath in self.map:
            return self.map[filepath]

        data = self._get_examples_from_file(filepath, 'train')
        print('Done')

        return data

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        filepath = Path(self.data_dir) / self.dev_file
        print('Getting dev examples')
        if filepath in self.map:
            return self.map[filepath]

        return self._get_examples_from_file(filepath, 'test')



class CsvClassifierDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: CsvClassifierDataTrainingArguments
    features: List[InputFeatures]

    def __init__(
            self,
            processor : CsvClassifierProcessor,
            args: CsvClassifierDataTrainingArguments,
            tokenizer: PreTrainedTokenizer,
            limit_length: Optional[int] = None,
            start_length: Optional[int] = 0,
            mode: Union[str, Split] = Split.train,
            cache_dir: Optional[str] = None,
        
    ):
        self.args = args
        print(f'Getting for {args.data_dir}')

        self.processor = processor
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError(f"mode {mode} is not a valid split name")


        logger.info(f"Creating features from dataset file at {args.data_dir}")

        if mode == Split.dev:
            examples = self.processor.get_dev_examples()
        else:
            examples = self.processor.get_train_examples()
        if limit_length is not None:
            examples = examples[start_length:limit_length]
        self.features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=100,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=256,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    print('Running convert_examples_to_features')
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples)):
        len_examples = len(examples)
        if ex_index % 1000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(text=example.text_a, text_pair=example.text_b, add_special_tokens=True,
                                       max_length=max_length, return_token_type_ids=True,
                                       )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(f'{tokenizer.decode(input_ids)}')
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % (example.label))
            logger.info(f'{tokenizer.decode(input_ids)}')
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, #token_type_ids=token_type_ids,
                label=example.label
            )
        )
    print('Done')
    return features

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import compute_class_weight

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, AutoModel
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='distilbert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_type: str = field(
        default="distilbert", metadata={"help": "Type of the model (e.g. bert-base-uncased)"}
    )

parser = HfArgumentParser((ModelArguments, CsvClassifierDataTrainingArguments, TrainingArguments))


training_args = TrainingArguments('/kaggle/working', save_total_limit=1)

model_args = ModelArguments()
data_args = CsvClassifierDataTrainingArguments()



# Set seed
set_seed(training_args.seed)

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=data_args.num_labels,
    cache_dir=data_args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=data_args.cache_dir,
)


processor = CsvClassifierProcessor(data_args.data_dir, train_file='train.csv', dev_file='test.csv',
                                                 source_key=data_args.source_key, label_key=data_args.target_key,
                                                 source_key_b=data_args.source_key_b)

# Get datasets
train_dataset = (
    CsvClassifierDataset(processor, data_args, tokenizer=tokenizer, cache_dir=data_args.cache_dir, limit_length=900000)
)


labels = [feature.label for feature in train_dataset]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

eval_dataset = (
    CsvClassifierDataset(processor, data_args, tokenizer=tokenizer, cache_dir=data_args.cache_dir, start_length=900000, limit_length=105000)
)


#save the processed dataset 

import pickle
pickle.dump(train_dataset, open('/kaggle/working/dataset.pkl', 'wb'))
!ls /kaggle/working
print(training_args)
training_args.per_device_train_batch_size = 128
training_args.num_train_epochs = 1
training_args.logging_steps = 100


model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=data_args.cache_dir,
)
model = model.to('cuda')

def compute_metrics_fn(p: EvalPrediction):
    prediction = p.predictions.argmax(axis=1)
    breakpoint()
    precision, recall, fbeta, *_ = precision_recall_fscore_support(y_true=p.label_ids, y_pred=prediction,
                                                                   average='weighted'
                                                                   )
    return {'accuracy': (prediction == p.label_ids).mean(),
            'precision': precision,
            'recall': recall,
            'f1': fbeta
            }

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics_fn,
)

# Training
trainer.train(
    model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
)
trainer.save_model()
# For convenience, we also re-save the tokenizer to the same directory,
# so that you can share your model easily on huggingface.co/models =)
if trainer.is_world_master():
    tokenizer.save_pretrained(training_args.output_dir)

# Evaluation
eval_results = {}
logger.info("*** Evaluate ***")

trainer.compute_metrics = compute_metrics_fn
eval_result = trainer.evaluate(eval_dataset=eval_dataset)

output_eval_file = os.path.join(
    training_args.output_dir, f"eval_results.txt"
)
if trainer.is_world_master():
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****")
        for key, value in eval_result.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

    eval_results.update(eval_result)



trainer.save_model()

df3 = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
df3.shape
processor2 = CsvClassifierProcessor(data_args.data_dir, train_file='test.csv', dev_file='test.csv',
                                                 source_key=data_args.source_key, label_key=data_args.target_key,
                                                 source_key_b=data_args.source_key_b)

test_dataset = (
    CsvClassifierDataset(processor2, data_args, tokenizer=tokenizer, mode=Split.dev, cache_dir=data_args.cache_dir)
)

len(test_dataset)

predictions = trainer.predict(test_dataset)
print('Done')
type(predictions[0])
predictions_softmax = torch.nn.functional.softmax(torch.tensor(predictions[0]))

predictions_softmax.shape
class_predictions = predictions_softmax.argmax(dim=1)
df3['prediction'] = class_predictions
df3.head()
del(df3['question_text'])
df3.to_csv('submission.csv', index=False)
print('Done')
