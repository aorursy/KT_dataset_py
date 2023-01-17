!pip install fast-bert
!git clone https://github.com/NVIDIA/apex
%cd apex

!pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

%cd ..
import pandas as pd

import os
!mkdir data

DATA_FOLDER = 'data'
train_df = pd.read_csv('../input/train.csv', index_col='id')

val_df = pd.read_csv('../input/valid.csv', index_col='id')

test_df = pd.read_csv('../input/test.csv', index_col='id') 
label_cols = list(pd.get_dummies(train_df['label']).columns)

with open(os.path.join(DATA_FOLDER, 'labels.csv'), 'w') as f:

    f.write('\n'.join(label_cols))

def to_fastbert(df:pd.DataFrame, name:str):

  d = pd.get_dummies(df['label'])

  d['text'] = df['text']

  d.to_csv(os.path.join(DATA_FOLDER, f'{name}.csv'), index=True, index_label=['id'])

  
for x,n in zip([train_df, val_df], ['train', 'val']):

  to_fastbert(x, n)
#!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
#!unzip multi_cased_L-12_H-768_A-12.zip
#!git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
#!python pytorch-pretrained-BERT/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path multi_cased_L-12_H-768_A-12/bert_model.ckpt --bert_config_file multi_cased_L-12_H-768_A-12/bert_config.json --pytorch_dump_path multi_cased_L-12_H-768_A-12/pytorch_model.bin
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pytorch_pretrained_bert.modeling import BertForPreTraining, BertConfig, BertForMaskedLM, BertForSequenceClassification

from pathlib import Path

import torch



from fastai.text import Tokenizer, Vocab

import pandas as pd

import collections

import os

from tqdm import tqdm, trange

import sys

import random

import numpy as np

import apex

from sklearn.model_selection import train_test_split



import datetime

    

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.optimization import BertAdam



from fast_bert.modeling import BertForMultiLabelSequenceClassification

from fast_bert.data import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features

from fast_bert.learner import BertLearner

from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc
torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)

run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
DATA_PATH = Path('data/')

LABEL_PATH = Path('data/')



MODEL_PATH=Path('models/')

LOG_PATH=Path('logs/')

MODEL_PATH.mkdir(exist_ok=True)



model_state_dict = None



FINETUNED_PATH = None

# model_state_dict = torch.load(FINETUNED_PATH)



LOG_PATH.mkdir(exist_ok=True)
# If using a local version, download and convert by uncommenting cells 10,11,12,13

#BERT_PRETRAINED_PATH = Path('multi_cased_L-12_H-768_A-12/')



# We'll let the library get a plus-n-play model 

BERT_PRETRAINED_PATH = 'bert-base-multilingual-cased'



args = {

    "run_text": "amazon pet review",

    "train_size": -1,

    "val_size": -1,

    "log_path": LOG_PATH,

    "full_data_dir": DATA_PATH,

    "data_dir": DATA_PATH,

    "task_name": "Amazon Pet Review",

    "no_cuda": False,

    "bert_model": BERT_PRETRAINED_PATH,

    "output_dir": MODEL_PATH/'output',

    "max_seq_length": 512,

    "do_train": True,

    "do_eval": True,

    "do_lower_case": False,

    "train_batch_size": 16,

    "eval_batch_size": 16,

    "learning_rate": 5e-6,

    "num_train_epochs": 4.0,

    "warmup_proportion": 0.1,

    "no_cuda": False,

    "local_rank": -1,

    "seed": 42,

    "gradient_accumulation_steps": 1,

    "optimize_on_cpu": False,

    "fp16": True,

    "loss_scale": 128

}
import logging



logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))



logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',

    datefmt='%m/%d/%Y %H:%M:%S',

    handlers=[

        logging.FileHandler(logfile),

        logging.StreamHandler(sys.stdout)

    ])



logger = logging.getLogger()
logger.info(args)
tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH, do_lower_case=args['do_lower_case'])
device = torch.device('cuda')

if torch.cuda.device_count() > 1:

    multi_gpu = True

else:

    multi_gpu = False
databunch = BertDataBunch(args['data_dir'], LABEL_PATH, tokenizer, train_file='train.csv', val_file='val.csv',

                          test_data=list(test_df['text'].values),

                          text_col="text", label_col=label_cols,

                          bs=args['train_batch_size'], maxlen=args['max_seq_length'], 

                          multi_gpu=multi_gpu, multi_label=True)

databunch.save()
num_labels = len(databunch.labels)
from functools import partial



metrics = []

metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})

metrics.append({'name': 'roc_auc', 'function': roc_auc})

metrics.append({'name': 'F1', 'function': partial(fbeta, beta=1)})

metrics.append({'name': 'accuracy_single', 'function': accuracy_multilabel})
learner = BertLearner.from_pretrained_model(databunch, BERT_PRETRAINED_PATH, metrics, device, logger, 

                                            finetuned_wgts_path=FINETUNED_PATH, 

                                            is_fp16=args['fp16'], loss_scale=args['loss_scale'], 

                                            multi_gpu=multi_gpu,  multi_label=True)
learner.fit(4, lr=args['learning_rate'], schedule_type="warmup_cosine_hard_restarts")
preds = learner.predict_batch()
test_df['label'] = [max(x, key=lambda z: z[1])[0] for x in preds]

test_df['label'].to_csv('fast_bert_submission.csv', index=True, index_label=['id'], header=True)
!rm -Rf apex