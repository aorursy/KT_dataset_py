!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import os

os.environ['XLA_USE_BF16'] = "1"

os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'



import torch

import pandas as pd

import numpy as np

import joblib

import logging

import transformers

import sys

import torch.nn as nn

import gc;

import h5py

from scipy import stats

from collections import OrderedDict, namedtuple

from torch.optim import lr_scheduler

from transformers import (

    AdamW, get_linear_schedule_with_warmup, get_constant_schedule, 

    XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,

)

from sklearn import metrics, model_selection

from tqdm.autonotebook import tqdm
import warnings

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
class CustomRoberta(nn.Module):

    def __init__(self, model, hid_mix=6):

        super(CustomRoberta, self).__init__()

        

        self.num_labels = 1

        self.hid_mix = hid_mix

        self.roberta = transformers.XLMRobertaModel.from_pretrained(model, 

                                                                    output_hidden_states=True, 

                                                                    num_labels=self.num_labels

                                                                   )

        self.classifier = nn.Linear(self.roberta.pooler.dense.out_features, self.num_labels)

        self.dropout = nn.Dropout(p=0.1)



    def forward(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None):

        

        outputs = self.roberta(input_ids, 

                               attention_mask=attention_mask,

                               position_ids=position_ids,

                               head_mask=head_mask,

                               inputs_embeds=inputs_embeds

                              )

        

        hidden_states = outputs[2]

        feats = self.roberta.pooler.dense.out_features

        

        hmix = []

        

        for i in range(1, self.hid_mix + 1):

            hmix.append(hidden_states[-i][:, 0].reshape((-1, 1, feats)))

        

        hmix_tensor = torch.cat(hmix, 1)

        mean_tensor = torch.mean(hmix_tensor, 1)

        pool_tensor = self.dropout(mean_tensor)

        

        return self.classifier(pool_tensor)
model = CustomRoberta(model="xlm-roberta-base", hid_mix=6);
class MyIterableDataset(torch.utils.data.Dataset):

        # np.array(x_train).shape, np.array(x_valid).shape -> ((435712, 3), (8000, 3))    

    

    def __init__(self, data_memmap, target_memmap, shape=()):

        self.data = np.memmap(data_memmap, shape=shape, mode="r", dtype="int32")

        self.target = np.memmap(target_memmap, shape=(shape[1],), mode="r", dtype="int32")

        self.shape = shape

    

    def __len__(self):

        return self.shape[1]

    

    def __getitem__(self, idx):

        # mem-map contains input_ids, masks, targets in that index order;

        return np.array(self.data[0][idx]), np.array(self.data[1][idx]), np.array(self.target[idx])
from pathlib import Path



root_path = Path("../input/memmap-tpu-xlmr-pytorch-pad-on-fly/")



train_dataset = MyIterableDataset(data_memmap = root_path / "train.mymemmap",

                                  target_memmap = root_path / "train_targets.mymemmap",

                                  shape = (2, 435712, 128),

                                 )



valid_dataset = MyIterableDataset(data_memmap = root_path / "valid.mymemmap",

                                  target_memmap = root_path / "valid_targets.mymemmap",

                                  shape = (2, 8000, 128),

                                )

gc.collect();
def run(ordinal):

    

    gc.collect();

    def loss_fn(outputs, targets):

        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))



    def train_loop_fn(data_loader, model, optimizer, device, scheduler=None, epoch=None):

        

        model.train()

        

        for bi, d in enumerate(data_loader):

            

            ids, mask, targets = d[0], d[1], d[2]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)



            optimizer.zero_grad()

            outputs = model(input_ids=ids, attention_mask=mask,)

            

            loss = loss_fn(outputs, targets)

            

            if bi % 25 == 0:

                xm.master_print(f'bi={bi}, loss={loss}')



            loss.backward()

            xm.optimizer_step(optimizer)

            

            if scheduler is not None:

                scheduler.step()

        

        model.eval();

        xm.save(model.state_dict(), f"xlm_roberta_large_model_{epoch}.bin")

        

    def eval_loop_fn(data_loader, model, device):

        

        model.eval()

        fin_targets = []

        fin_outputs = []

        for bi, d in enumerate(data_loader):

            

            if bi % 25 == 0:

                xm.master_print(f'EVAL bi={bi}')

            

            ids, mask, targets = d[0], d[1], d[2]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)



            outputs = model(input_ids=ids, attention_mask = mask,)



            targets_np = targets.cpu().detach().numpy().tolist()

            outputs_np = outputs.cpu().detach().numpy().tolist()

            fin_targets.extend(targets_np)

            fin_outputs.extend(outputs_np)    



        return fin_outputs, fin_targets

    

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 16

    EPOCHS = 2 # change



    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')



    train_sampler = torch.utils.data.distributed.DistributedSampler(

          train_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=True

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=4,

    )



    valid_sampler = torch.utils.data.distributed.DistributedSampler(

          valid_dataset,

          num_replicas=xm.xrt_world_size(),

          rank=xm.get_ordinal(),

          shuffle=False,

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=1,

    )



    device = xm.xla_device();

    model.to(device);

    

    lr = 1e-4 * xm.xrt_world_size()

    

    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)

    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')



    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = None



    for epoch in range(EPOCHS):

        gc.collect();

        para_loader = pl.ParallelLoader(train_data_loader, [device])

        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler, epoch=epoch)

        

        para_loader = pl.ParallelLoader(valid_data_loader, [device])

        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)

        

        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)

        del o, t

        

        try:

            print(f'{ordinal} AUC={auc}')

        except:

            pass

        

        def reduce_fn(vals):

            return sum(vals) / len(vals)



        auc = xm.mesh_reduce('auc_reduce', auc, reduce_fn)

        xm.master_print('AUC={:.4f}'.format(auc))

        gc.collect();
%%time



def _mp_fn(rank, flags):

    a = run(rank)



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')