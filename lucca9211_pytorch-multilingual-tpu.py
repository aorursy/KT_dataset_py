!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.utils import shuffle

from sklearn import metrics

# imports pytorch

import torch

import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader

# imports the torch_xla package

import torch_xla

import torch_xla.distributed.parallel_loader as pl

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp





#transformers

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings



warnings.filterwarnings("ignore")
# # Check is TPU is working with Pytorch

# dev = xm.xla_device()

# t1 = torch.ones(8, 3, device = dev)

# print(t1)
BERT_ROOT = "../input/bertbasemultilingualuncased/"
DATA_ROOT = Path("..")/"input"/ "jigsaw-multilingual-toxic-comment-classification/"



df1,df2,df3,test,sample = [pd.read_csv(DATA_ROOT / fname) for fname in ["jigsaw-toxic-comment-train.csv",

                                                                        "jigsaw-unintended-bias-train.csv",

                                                                        "validation.csv",

                                                                        "test.csv",

                                                                        "sample_submission.csv"

                                                                       ]]

df2.toxic = df2.toxic.round().astype(int)

train = pd.concat([

    df1[['comment_text', 'toxic']],

    df3[['comment_text', 'toxic']],

    df2[['comment_text', 'toxic']].query('toxic==1'),

    df2[['comment_text', 'toxic']].query('toxic==0').sample(n=200000, random_state=0)

])



valid = df3
train = train.dropna()



train = shuffle(train, random_state=22)

train.head()
valid = valid.dropna()



valid = shuffle(valid, random_state=22)

valid.head()
class DatasetClass:

    def __init__(self, text, labels, tokenizer, max_length):

        self.text = text

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.labels = labels



    def __len__(self):

        return len(self.text)



    def __getitem__(self, item):

        text = str(self.text[item])

        text = " ".join(text.split())



        inputs = self.tokenizer.encode_plus(

            text,

            None,

            add_special_tokens=True,

            max_length=self.max_length,

        )

        ids = inputs["input_ids"]

        token_type_ids = inputs["token_type_ids"]

        mask = inputs["attention_mask"]



        padding_length = self.max_length - len(ids)



        ids = ids + ([0] * padding_length)

        mask = mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)



        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),

            'labels': torch.tensor(self.labels[item], dtype=torch.float)

        }





class Bert_fn(nn.Module):

    def __init__(self, bert_path):

        super(Bert_fn, self).__init__()

        self.bert_path = bert_path

        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.bert_drop = nn.Dropout(0.3)

        self.out = nn.Linear(768 * 2, 1)



    def forward(

            self,

            ids,

            mask,

            token_type_ids

    ):

        o1, o2 = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids)



        apool = torch.mean(o1, 1)

        mpool, _ = torch.max(o1, 1)

        cat = torch.cat((apool, mpool), 1)



        bo = self.bert_drop(cat)

        p2 = self.out(bo)

        return p2



BERT_MODEL = Bert_fn(bert_path="../input/bertbasemultilingualuncased/")
def main_fn():

    def loss_fn(outputs, labels):

        return nn.BCEWithLogitsLoss()(outputs, labels.view(-1, 1))



    def train_fn(data_loader, model, optimizer, device, scheduler=None):

        model.train()

        for bi, d in enumerate(data_loader):

            ids = d["ids"]

            mask = d["mask"]

            token_type_ids = d["token_type_ids"]

            labels = d["labels"]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            labels = labels.to(device, dtype=torch.float)



            optimizer.zero_grad()

            outputs = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )



            loss = loss_fn(outputs, labels)

            if bi % 10 == 0:

                xm.master_print(f'bi={bi}, loss={loss}')



            loss.backward()

            xm.optimizer_step(optimizer)

            if scheduler is not None:

                scheduler.step()



    def eval_fn(data_loader, model, device):

        model.eval()

        fin_targets = []

        fin_outputs = []

        for bi, d in enumerate(data_loader):

            ids = d["ids"]

            mask = d["mask"]

            token_type_ids = d["token_type_ids"]

            labels = d["labels"]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            labels = labels.to(device, dtype=torch.float)



            outputs = model(

                ids=ids,

                mask=mask,

                token_type_ids=token_type_ids

            )



            targets_np = labels.cpu().detach().numpy().tolist()

            outputs_np = outputs.cpu().detach().numpy().tolist()

            fin_targets.extend(targets_np)

            fin_outputs.extend(outputs_np)



        return fin_outputs, fin_targets



    MAX_LEN = 192

    BATCH_SIZE = 64

    EPOCHS = 2



    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_ROOT, do_lower_case=True)



    train_labels = train.toxic.values

    valid_labels = valid.toxic.values



    train_dataset = DatasetClass(

        text=train.comment_text.values,

        labels=train_labels,

        tokenizer=tokenizer,

        max_length=MAX_LEN

    )



    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True)



    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=4

    )



    valid_dataset = DatasetClass(

        text=valid.comment_text.values,

        labels=valid_labels,

        tokenizer=tokenizer,

        max_length=MAX_LEN

    )



    valid_sampler = torch.utils.data.distributed.DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=False)



    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=64,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=4

    )



    device = xm.xla_device()

    model = BERT_MODEL.to(device)



    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(

            nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



    lr = 0.4 * 1e-5 * xm.xrt_world_size()

    num_train_steps = int(len(train_dataset) /

                          BATCH_SIZE / xm.xrt_world_size() * EPOCHS)

    xm.master_print(

        f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')



    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )



    for epoch in range(EPOCHS):

        para_loader = pl.ParallelLoader(train_loader, [device])

        train_fn(para_loader.per_device_loader(device), model,

                 optimizer, device, scheduler=scheduler)



        para_loader = pl.ParallelLoader(valid_loader, [device])

        o, t = eval_fn(para_loader.per_device_loader(device), model, device)

        xm.save(model.state_dict(), "model.bin")

        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)

        xm.master_print(f'AUC = {auc}')

def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = main_fn()





FLAGS = {}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


