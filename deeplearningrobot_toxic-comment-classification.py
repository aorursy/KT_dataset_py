# Install PyTorch/XLA
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import os
os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
import sys
from sklearn import metrics, model_selection

import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

warnings.filterwarnings("ignore")
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# Class to create datasets from numpy arrays
class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self,*arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
        self.arrays = arrays
    
    def __getitem__(self, index):
        return tuple(torch.from_numpy(np.array(array[index])) for array in self.arrays)
    
    def __len__(self):
        return self.arrays[0].shape[0]
tokenized_path = '../input/comments-preprocessed/'
x_train = np.load(tokenized_path+'x_train.npy',mmap_mode='r')
train_toxic = np.load(tokenized_path+'df_train_toxic.npy',mmap_mode='r')

x_valid = np.load(tokenized_path+'x_valid.npy',mmap_mode='r')
valid_toxic = np.load(tokenized_path+'df_valid_toxic.npy',mmap_mode='r')

x_train.shape, x_valid.shape
train_dataset = ArrayDataset(x_train, train_toxic)
valid_dataset = ArrayDataset(x_valid, valid_toxic)
# Delete unused variables
del x_train, x_valid
import gc;gc.collect()
gc.collect()
class CustomTransformer(nn.Module):
    def __init__(self):
        super(CustomTransformer, self).__init__()
        self.num_labels = 1
        self.roberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-large", output_hidden_states=False, num_labels=1) # Choose a model from https://huggingface.co/transformers/pretrained_models.html
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):

        _, o2 = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        logits = self.classifier(o2)       
        outputs = logits
        return outputs
    
mx = CustomTransformer();
mx
import torch_xla.version as xv
print('PYTORCH:', xv.__torch_gitrev__)
print('XLA:', xv.__xla_gitrev__)
!free -h
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
def reduce_fn(vals):
    return sum(vals) / len(vals)
def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    
    for bi, d in enumerate(data_loader):
        ids = d[0]
        targets = d[1]

        ids = ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            input_ids=ids,
        )
        loss = loss_fn(outputs, targets)
        if bi % 50 == 0:
            loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn)
            xm.master_print(f'bi={bi}, loss={loss_reduced}')
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()
            
    model.eval()
    
def eval_loop_fn(data_loader, model, device):
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        ids = d[0]
        targets = d[1]

        ids = ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            input_ids=ids,
        )

        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)    
        del targets_np, outputs_np
        gc.collect()
    return fin_outputs, fin_targets
def _run():
    MAX_LEN = 192
    TRAIN_BATCH_SIZE = 16
    EPOCHS = 1

    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
    )
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=4,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=0
    )

    device = xm.xla_device()
    model = mx.to(device)
    xm.master_print('The model is loaded onto the device')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 0.5e-5 * xm.xrt_world_size()
    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')


    for epoch in tqdm(range(EPOCHS)):
        gc.collect()
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        xm.master_print('Parallel loader created... Training...')
        gc.collect()
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)
        del para_loader
        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        gc.collect()
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        del para_loader
        gc.collect()
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        auc_reduced = xm.mesh_reduce('auc_reduce',auc,reduce_fn)
        xm.master_print(f'AUC = {auc_reduced}')
        gc.collect()
    xm.save(model.state_dict(), "xlm_roberta_model.bin")
import time

# Start training processes
def _mp_fn(rank, flags):
    a = _run()

FLAGS={}
start_time = time.time()
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

print('Time taken: ',time.time()-start_time)