!pip install -U catalyst==19.11.1 transformers > /dev/null
# Python 

import os

import warnings

import logging

from typing import Mapping, List

from pprint import pprint



# Numpy and Pandas 

import numpy as np

import pandas as pd



# PyTorch 

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader



# Transformers 

from transformers import AutoConfig, AutoModel, AutoTokenizer



# Catalyst

from catalyst.dl import SupervisedRunner

from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback, OptimizerCallback

from catalyst.dl.callbacks import CheckpointCallback, InferCallback

from catalyst.utils import set_global_seed, prepare_cudnn
MODEL_NAME = 'distilbert-base-uncased' # pretrained model from Transformers

LOG_DIR = "./logdir"                   # for training logs and tensorboard visualizations

NUM_EPOCHS = 3                         # smth around 2-6 epochs is typically fine when finetuning transformers

BATCH_SIZE = 8                        # depends on your available GPU memory (in combination with max seq length)

MAX_SEQ_LENGTH = 512                   # depends on your available GPU memory (in combination with batch size)

NUM_CLASSES = 3                        # solving 3-class classification problem

LEARN_RATE = 5e-5                      # learning rate is typically ~1e-5 for transformers

ACCUM_STEPS = 4                        # one optimization step for that many backward passes

SEED = 17                              # random seed for reproducibility
FP16_PARAMS = None
%%capture

# if Your machine doesn't support FP16, comment this 3 lines below

!git clone https://github.com/NVIDIA/apex 

!pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex 

!rm -rf ./apex

FP16_PARAMS = dict(opt_level="O1") 
# to reproduce, download the data and customize this path

PATH_TO_DATA = '../input/clickbait-news-detection//'
train_df = pd.read_csv(PATH_TO_DATA + 'train.csv', index_col='id').fillna('')

valid_df = pd.read_csv(PATH_TO_DATA + 'valid.csv', index_col='id').fillna('')

test_df = pd.read_csv(PATH_TO_DATA + 'test.csv', index_col='id').fillna('')
train_df.head()
# target distribution

train_df['label'].value_counts(normalize=True)
# statistics of text length (in words)

train_df['text'].apply(lambda s: len(s.split())).describe()
train_df['title_n_text'] = train_df['title'] + '_' + train_df['text']

valid_df['title_n_text'] = valid_df['title'] + '_' + valid_df['text']

test_df['title_n_text'] = test_df['title'] + '_' + test_df['text']
class TextClassificationDataset(Dataset):

    def __init__(self, 

                 texts: List, 

                 labels: List=None, 

                 label_dict: Mapping[str, int]=None,

                 max_seq_length: int=512,

                 model_name: str='distilbert-base-uncased'):

        '''

        :param texts: a list with texts to classify or to train the classifier on

        :param labels: a list with classification labels, can be strings as well (optional)

        :param label_dict: a dictionary mapping class names to class ids, to be passed to the validation data

        :param max_seq_length: maximal sequence length, texts will be stripped 

        :model_name: transformer model name, we need here to perform appropriate tokenization

        '''

        self.texts = texts

        self.labels = labels

        self.label_dict = label_dict

        

        if self.label_dict is None and labels is not None:

            # {'class1': 0, 'class2': 1, 'class3': 2, ...}

            # using this instead of `sklearn.preprocessing.LabelEncoder`

            # no easily handle unknown target values

            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        

        self.max_seq_length = max_seq_length



        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # suppresses tokenizer warnings

        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

        

        # special tokens for transformers 

        # in the simplest case a [CLS] token is added in the beginning

        # and [SEP] token is added in the end of a piece of text

        self.sep_vid = self.tokenizer.vocab["[SEP]"]

        self.cls_vid = self.tokenizer.vocab["[CLS]"]

        self.pad_vid = self.tokenizer.vocab["[PAD]"]



    def __len__(self):

        return len(self.texts)



    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:



        x = self.texts[index]



        # encoding the text

        x_encoded = self.tokenizer.encode(

            x,

            add_special_tokens=True,

            max_length=self.max_seq_length,

            return_tensors="pt",

        ).squeeze(0)

        

        # padding short texts

        true_seq_length = x_encoded.size(0)

        pad_size = self.max_seq_length - true_seq_length

        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()

        x_tensor = torch.cat((x_encoded, pad_ids))

        

        # encoding target

        if self.labels is not None:

            y = self.labels[index]

            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)

        

        # dealing with masks

        mask = torch.ones_like(x_encoded, dtype=torch.int8)

        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)

        mask = torch.cat((mask, mask_pad))

        

        output_dict = {

            "features": x_tensor,

            'attention_mask': mask

        }

        

        if self.labels is not None:

            output_dict["targets"] = y_encoded



        return output_dict
train_dataset = TextClassificationDataset(

    texts=train_df['text'],

    labels=train_df['label'],

    label_dict=None,

    max_seq_length=MAX_SEQ_LENGTH)



valid_dataset = TextClassificationDataset(

    texts=valid_df['text'],

    labels=valid_df['label'],

    label_dict=train_dataset.label_dict,

    max_seq_length=MAX_SEQ_LENGTH)



test_dataset = TextClassificationDataset(

    texts=test_df['text'],

    labels=None,

    label_dict=None,

    max_seq_length=MAX_SEQ_LENGTH)
train_df.loc[1]
pprint(train_dataset[1])
train_val_loaders = {

    "train": DataLoader(dataset=train_dataset,

                        batch_size=BATCH_SIZE, 

                        shuffle=True),

    "valid": DataLoader(dataset=valid_dataset,

                        batch_size=BATCH_SIZE, 

                        shuffle=False)    

}
class DistilBertForSequenceClassification(nn.Module):

    def __init__(self, model_name, num_classes=None):

        super().__init__()

        

        config = AutoConfig.from_pretrained(

            model_name, num_labels=num_classes)

        

        self.distilbert = AutoModel.from_pretrained(model_name, 

                                                    config=config)

        self.pre_classifier = nn.Linear(config.dim, config.dim)

        self.classifier = nn.Linear(config.dim, num_classes)

        self.dropout = nn.Dropout(config.seq_classif_dropout)



    def forward(self, features, attention_mask=None, head_mask=None):

        assert attention_mask is not None, "attention mask is none"

        distilbert_output = self.distilbert(input_ids=features,

                                            attention_mask=attention_mask,

                                            head_mask=head_mask)

        hidden_state = distilbert_output[0]                  # (bs, seq_len, dim)

        pooled_output = hidden_state[:, 0]                   # (bs, dim)

        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)

        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)

        pooled_output = self.dropout(pooled_output)          # (bs, dim)

        logits = self.classifier(pooled_output)              # (bs, dim)



        return logits
model = DistilBertForSequenceClassification(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # can be changed in case of multiple GPUs onboard

set_global_seed(SEED)                       # reproducibility

prepare_cudnn(deterministic=True)           # reproducibility
# we need a small wrapper around Catalyst's runner to be able to pass masks to it

class BertSupervisedRunner(SupervisedRunner):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, input_key=(

            "features",

            "attention_mask",

        ), **kwargs)
# model runner

runner = BertSupervisedRunner()



# model training

runner.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler,

    loaders=train_val_loaders,

    callbacks=[

        AccuracyCallback(num_classes=NUM_CLASSES),

#         F1ScoreCallback(activation='Softmax'), # throws a tensor shape mismatch error

        OptimizerCallback(accumulation_steps=ACCUM_STEPS)

    ],

    fp16=FP16_PARAMS,

    logdir=LOG_DIR,

    num_epochs=NUM_EPOCHS,

    verbose=True

)
!nvidia-smi
torch.cuda.empty_cache()
!nvidia-smi
from catalyst.dl.utils import plot_metrics
plot_metrics(

    logdir=LOG_DIR,

    step='batch',

    metrics=['loss', 'accuracy01']

)
test_loaders = {

    "test": DataLoader(dataset=test_dataset,

                        batch_size=BATCH_SIZE, 

                        shuffle=False) 

}
runner.infer(

    model=model,

    loaders=test_loaders,

    callbacks=[

        CheckpointCallback(

            resume=f"{LOG_DIR}/checkpoints/best.pth"

        ),

        InferCallback(),

    ],   

    verbose=True

)
predicted_probs = runner.callbacks[0].predictions['logits']
sample_sub_df = pd.read_csv(PATH_TO_DATA + 'sample_submission.csv',

                           index_col='id')
train_dataset.label_dict
sample_sub_df['label'] = predicted_probs.argmax(axis=1)

sample_sub_df['label'] = sample_sub_df['label'].map({v:k for k, v in train_dataset.label_dict.items()})
sample_sub_df.head()
sample_sub_df.to_csv('distillbert_submission.csv')