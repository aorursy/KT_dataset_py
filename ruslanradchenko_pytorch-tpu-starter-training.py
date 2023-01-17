!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os.path

import os

import shutil

import sys

import time

from datetime import datetime

import glob



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

import torch

import torch.nn as nn

from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler

from transformers import *
import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils

import warnings



warnings.filterwarnings("ignore")
home_dir = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

df = pd.read_csv(os.path.join(home_dir, "jigsaw-toxic-comment-train-processed-seqlen128.csv"))

df
df_train = df

df_val = pd.read_csv(os.path.join(home_dir, "validation-processed-seqlen128.csv"))

df_test = pd.read_csv(os.path.join(home_dir, "test-processed-seqlen128.csv"))
df_train["toxic"].hist()

plt.plot()
batch_size = 32
def str_to_t(s):

    return torch.tensor(np.array(s[1:-1].split(',')).astype(np.int32))



class ToxicDataset(Dataset):

    def __init__(self, df):

        self.df = df

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        try:

            label = row['toxic']

        except Exception:

            label = -1 #test dataset

        return {

            'id': idx,

            'input_ids': torch.tensor(str_to_t(row["input_word_ids"])),

            'mask': torch.tensor(str_to_t(row["input_mask"])),

            'label': label

        }
class MyBert(nn.Module):    

    def __init__(self):

        super(MyBert, self).__init__()

        self.bm = BertModel.from_pretrained("bert-base-multilingual-cased")

        self.do = nn.Dropout(p=0.2)

        self.fc = nn.Linear(768 * 2, 1)



    def forward(self, input_ids, attention_mask):

        x = self.bm(input_ids=input_ids, attention_mask=attention_mask)[0]

        mx, _ = torch.max(x, 1)

        mean = torch.mean(x, 1)

        x = torch.cat((mx, mean), 1)

        x = self.do(x)

        x = self.fc(x)

        return x[:, 0]

    

model = MyBert()
def clear_dir(d):

    folder = d

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        try:

            if os.path.isfile(file_path) or os.path.islink(file_path):

                os.unlink(file_path)

            elif os.path.isdir(file_path):

                shutil.rmtree(file_path)

        except Exception as e:

            print('Failed to delete %s. Reason: %s' % (file_path, e))



try:

    os.mkdir('./tmp')

except Exception:

    clear_dir('./tmp')
def _run():

    device = xm.xla_device()

    train_dataset = ToxicDataset(df_train)

    val_dataset = ToxicDataset(df_val)



    train_sampler = DistributedSampler(

              train_dataset,

              num_replicas=xm.xrt_world_size(),

              rank=xm.get_ordinal(),

        )



    val_sampler = DistributedSampler(

              val_dataset,

              num_replicas=xm.xrt_world_size(),

              rank=xm.get_ordinal(),

        )



    train_dataloader = DataLoader(

            train_dataset,

            batch_size=batch_size,

            sampler=train_sampler,

            num_workers=1,

            drop_last=True

        )



    val_dataloader = DataLoader(

            val_dataset,

            batch_size=batch_size,

            sampler=val_sampler,

            num_workers=1,

            drop_last=True

        )

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]



    lr = 0.5e-5 * xm.xrt_world_size()

    epochs = 2

    num_train_steps = int(len(train_dataset) / batch_size / xm.xrt_world_size() * epochs)

    

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )

    for epoch in range(epochs):

        para_loader = pl.ParallelLoader(train_dataloader, [device]).per_device_loader(device)

        model.train()

        for bn, batch in enumerate(para_loader):

            if bn % 20 == 0:

                xm.master_print(f"Batch number {bn}/{len(para_loader)}")

            model.zero_grad()

            input_ids = batch["input_ids"].to(device).long()

            mask = batch["mask"].to(device).long()

            labels = batch["label"].to(device).float()

            outputs = model(

                input_ids, 

                attention_mask=mask, 

            )

            loss = criterion(outputs, labels)

            loss.backward()

            xm.master_print(f'Loss on batch {bn}/{len(para_loader)}: {loss.item()}')

            clip_grad_norm_(model.parameters(), 1.0)

            xm.optimizer_step(optimizer)

            if scheduler is not None:

                scheduler.step()

    xm.save(model.state_dict(), f"model.pt")

    xm.master_print("kek")

    '''model.eval()

    para_loader = pl.ParallelLoader(val_dataloader, [device]).per_device_loader(device)

    for bn, batch in enumerate(para_loader):

        if bn % 200 == 0:

            xm.master_print(f"Batch number {bn}/{len(para_loader)}")

        input_ids = batch["input_ids"].to(device)

        mask = batch["mask"].to(device)

        labels = batch["label"].detach().cpu().numpy()

        with torch.no_grad():

            logits = model(

                input_ids, 

                attention_mask=mask, 

            )[0][:, 1].detach().cpu().numpy()

        #auc = metrics.roc_auc_score(labels, logits)

        xm.master_print(f'AUC = {auc}')

    test_dataset = ToxicDataset(df_test)

    test_sampler = DistributedSampler(

          test_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

    )

test_dataloader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        sampler=test_sampler,

        num_workers=1,

        drop_last=True

    )

model.eval()

para_loader = pl.ParallelLoader(test_dataloader, [device]).per_device_loader(device)

xm.master_print("gonna print")

for bn, batch in enumerate(para_loader):

    input_ids = batch["input_ids"].to(device).long()

    mask = batch["mask"].to(device).long()

    answers = model(

            input_ids,

            attention_mask=mask

        )

    ids = batch["id"]

    ans_df = pd.Dataframe({'id': ids.numpy(), 'toxic': answers.numpy()})

    ans_df.to_csv(f"./tmp/sub_{datetime.utcnow().microsecond}.csv")

'''
def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    _run()



xmp.spawn(_mp_fn, args=({},), nprocs=8, start_method='fork')
del model

model = MyBert()

model.load_state_dict(torch.load('model.pt'))

device = xm.xla_device()

model.eval().to(device)

ans = []

test_dataset = ToxicDataset(df_test)

test_dataloader = DataLoader(

            test_dataset,

            batch_size=batch_size

        )



for batch in test_dataloader:

    input_ids = batch["input_ids"].to(device).long()

    mask = batch["mask"].to(device).long()

    outputs = nn.Sigmoid()(model(

        input_ids, 

        attention_mask=mask, 

    ))

    for i in range(outputs.size()[0]):

        ans.append(outputs[i].item())

xm.master_print("Fin")

ans_df = pd.DataFrame({"id": list(range(len(ans))), "toxic": ans})

ans_df.to_csv("submission.csv", index=False)