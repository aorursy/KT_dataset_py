!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os



import dataset

import engine

import torch

import transformers

import warnings



import pandas as pd

import numpy as np

import torch.nn as nn



from model import JigsawModel



from sklearn import metrics

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp



warnings.filterwarnings("ignore")
class config:

    MAX_LEN = 192

    TRAIN_BATCH_SIZE = 64

    VALID_BATCH_SIZE = 4

    EPOCHS = 1

    LEARNING_RATE = 0.5e-5

    BERT_PATH = "../input/bert-base-multilingual-uncased/"

    MODEL_PATH = "model.bin"

    TOKENIZER = transformers.BertTokenizer.from_pretrained(

        BERT_PATH,

        do_lower_case=True

    )

    JIGSAW_DATA_PATH = "../input/jigsaw-multilingual-toxic-comment-classification/"

    TRAINING_FILE_1 = os.path.join(

        JIGSAW_DATA_PATH, 

        "jigsaw-toxic-comment-train.csv"

    )

    TRAINING_FILE_2 = os.path.join(

        JIGSAW_DATA_PATH, 

        "jigsaw-unintended-bias-train.csv"

    )

    VALIDATION_FILE = os.path.join(

        JIGSAW_DATA_PATH, 

        "validation.csv"

    )
MX = JigsawModel(config.BERT_PATH)



df_train1 = pd.read_csv(

    config.TRAINING_FILE_1, 

    usecols=["comment_text", "toxic"]

).fillna("none")



df_train2 = pd.read_csv(

    config.TRAINING_FILE_2, 

    usecols=["comment_text", "toxic"]

).fillna("none")



df_valid = pd.read_csv(config.VALIDATION_FILE)



df_train = pd.concat([df_train1, df_train2], axis=0).reset_index(drop=True)

df_train = df_train.sample(frac=1).reset_index(drop=True).head(200000)



df_train = df_train.reset_index(drop=True)

df_valid = df_valid.reset_index(drop=True)



train_targets = df_train.toxic.values

valid_targets = df_valid.toxic.values
def run():

    train_dataset = dataset.JigsawTraining(

        comment_text=df_train.comment_text.values,

        targets=train_targets,

        config=config

    )



    train_sampler = torch.utils.data.distributed.DistributedSampler(

          train_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=True)



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=config.TRAIN_BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=2

    )



    valid_dataset = dataset.JigsawTraining(

        comment_text=df_valid.comment_text.values,

        targets=valid_targets,

        config=config

    )



    valid_sampler = torch.utils.data.distributed.DistributedSampler(

          valid_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=False)



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=config.VALID_BATCH_SIZE,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=1

    )



    device = xm.xla_device()

    model = MX.to(device)

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {

            'params': [

                p for n, p in param_optimizer if not any(

                    nd in n for nd in no_decay

                )

            ], 

            'weight_decay': 0.001

        },

        {

            'params': [

                p for n, p in param_optimizer if any(

                    nd in n for nd in no_decay

                )

            ],

            'weight_decay': 0.0

        },

    ]



    num_train_steps = int(

        len(df_train) / config.TRAIN_BATCH_SIZE / xm.xrt_world_size() * config.EPOCHS

    )

    optimizer = AdamW(

        optimizer_parameters, 

        lr=config.LEARNING_RATE * xm.xrt_world_size()

    )

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )



    best_auc = 0

    for epoch in range(config.EPOCHS):

        para_loader = pl.ParallelLoader(train_data_loader, [device])

        engine.train_fn(

            para_loader.per_device_loader(device), 

            model, 

            optimizer, 

            device, 

            scheduler

        )

        

        para_loader = pl.ParallelLoader(valid_data_loader, [device])

        outputs, targets = engine.eval_fn(

            para_loader.per_device_loader(device), 

            model, 

            device

        )



        targets = np.array(targets) >= 0.5

        auc = metrics.roc_auc_score(targets, outputs)

        print(f'[xla:{xm.get_ordinal()}]: AUC={auc}')

        if auc > best_auc:

            xm.save(model.state_dict(), config.MODEL_PATH)

            best_auc = auc
def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = run()
FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')