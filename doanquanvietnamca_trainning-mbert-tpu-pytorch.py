!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os
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
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection

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

train_enable = False

def one_hot_embedding(labels, num_classes=3):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 


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


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, 3)

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


class BERTDatasetTraining:
    def __init__(self,premise, hypothesis, targets, tokenizer, max_length):
        self.premise = premise
        self.hypothesis = hypothesis
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = targets

    def __len__(self):
        return len(self.premise)

    def __getitem__(self, item):
        hypothesis = str(self.hypothesis[item]+ self.premise[item])
        hypothesis = " ".join(hypothesis.split())

        inputs = self.tokenizer.encode_plus(
            hypothesis,
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
            'targets': torch.tensor(one_hot_embedding(self.targets[item]), dtype=torch.float)
        }
mx = BERTBaseUncased(bert_path="../input/bertbasemultilingualuncased")
df_train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv", usecols=["premise", "hypothesis", "label"]).fillna("none")
df_test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
df_train.head()
from sklearn.model_selection import train_test_split
df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=42)
def _run():
    def loss_fn(outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
        model.train()
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs, targets)
            if bi % 10 == 0:
                xm.master_print(f'bi={bi}, loss={loss}')

            loss.backward()
            xm.optimizer_step(optimizer)
            if scheduler is not None:
                scheduler.step()

    def eval_loop_fn(data_loader, model, device):
        model.eval()
        fin_targets = []
        fin_outputs = []
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            targets_np = targets.cpu().detach().numpy().tolist()
            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_targets.extend(targets_np)
            fin_outputs.extend(outputs_np)    

        return fin_outputs, fin_targets

    
    MAX_LEN = 192
    TRAIN_BATCH_SIZE = 64
    EPOCHS = 10

    tokenizer = transformers.BertTokenizer.from_pretrained("../input/bertbasemultilingualuncased", do_lower_case=True)

    train_targets = df_train.label.values
    valid_targets = df_valid.label.values

    train_dataset = BERTDatasetTraining(
        premise=df_train.premise.values,
        hypothesis = df_train.hypothesis.values,
        targets=train_targets,
        tokenizer=tokenizer,
        max_length=MAX_LEN
    )

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
        num_workers=4
    )

    valid_dataset = BERTDatasetTraining(
        premise=df_valid.premise.values,
        hypothesis = df_valid.hypothesis.values,
        targets=valid_targets,
        tokenizer=tokenizer,
        max_length=MAX_LEN
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=64,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=4
    )

    device = xm.xla_device()
    model = mx.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 0.4 * 1e-5 * xm.xrt_world_size()
    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    for epoch in range(EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)

        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        xm.save(model.state_dict(), "model.bin")
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        xm.master_print(f'Epochs: {epoch} AUC = {auc}')
# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run()

if train_enable:
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
MAX_LEN = 192
TEST_BATCH_SIZE = 64
class BERTDatasetTest:
    def __init__(self,premise, hypothesis, tokenizer, max_length):
        self.premise = premise
        self.hypothesis = hypothesis
        self.tokenizer = tokenizer
        self.max_length = max_length
     

    def __len__(self):
        return len(self.premise)

    def __getitem__(self, item):
        hypothesis = str(self.hypothesis[item]+ self.premise[item])
        hypothesis = " ".join(hypothesis.split())

        inputs = self.tokenizer.encode_plus(
            hypothesis,
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
            
        }
    
tokenizer = transformers.BertTokenizer.from_pretrained("../input/bertbasemultilingualuncased", do_lower_case=True)
test_dataset = BERTDatasetTest(
        premise=df_test.premise.values,
        hypothesis = df_test.hypothesis.values,
        tokenizer=tokenizer,
        max_length=MAX_LEN
    )

test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        sampler=test_sampler,
        drop_last=False,
        num_workers=4
    )

mx = BERTBaseUncased(bert_path="../input/bertbasemultilingualuncased")
device = 'xla:0'
model = mx.to(device)
chkp = torch.load("../input/mberttrain10epochs/model.bin")
mx.load_state_dict(chkp)
mx.eval()
fin_outputs = []
for bi, d in enumerate(test_data_loader):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids
                )
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_outputs.extend(outputs_np) 
preds = np.argmax(fin_outputs, axis=1)
df_submit = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
df_submit['prediction'] = preds
df_submit.head()
df_submit.to_csv("submission.csv", index=False)
fin_outputs1


fin_outputs1 = torch.nn.functional.softmax(torch.tensor(fin_outputs))
df_submit['prob0'] = fin_outputs1[:,0]
df_submit['prob1'] = fin_outputs1[:,1]
df_submit['prob2'] = fin_outputs1[:,2]
df_submit.to_csv('submission_prob_bert.csv', index=False)
df_submit.head()
