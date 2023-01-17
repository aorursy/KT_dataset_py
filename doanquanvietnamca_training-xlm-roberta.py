!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
!pip install --upgrade pip
!pip install --upgrade transformers==3.0.2
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
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, 3)

    def forward(
            self,
            ids,
            mask,
    ):
        o1, o2 = self.bert(
            ids,
            attention_mask=mask
           )
        
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)
        
        logits = torch.mean(torch.stack([
            #Multi Sample Dropout takes place here
            self.out(self.bert_drop(cat))
            for _ in range(5)
        ], dim=0), dim=0)
        return logits
mx = BERTBaseUncased()
df_train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv", usecols=["premise", "hypothesis", "label"]).fillna("none")
df_test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
df_train.head()
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
        hypothesis = str(self.hypothesis[item])
        premise = str(self.premise[item])

        hypothesis = " ".join(hypothesis.split())
        premise = " ".join(premise.split())
        hypothesis_inputs = self.tokenizer.encode_plus(
            hypothesis,
            None,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
            
        )
        premise_inputs = self.tokenizer.encode_plus(
            premise,
            None,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
            
        )
        ids = [0] + premise_inputs["input_ids"] + [2,2] + hypothesis_inputs['input_ids'] + [2]

        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
        else:
            ids = ids[:self.max_length-1]+ [2]
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(masks, dtype=torch.long),
            'targets': torch.tensor(one_hot_embedding(self.targets[item]), dtype=torch.float)
        }
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("xlm-roberta-base",do_lower_case=True)
train_dataset = BERTDatasetTraining(
        premise=df_train.premise.values,
        hypothesis = df_train.hypothesis.values,
        targets=df_train.label.values,
        tokenizer=tokenizer,
        max_length=192
    )

train_dataset[1]
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_train.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_train.index, y=df_train['label'])):
        df_train.loc[df_train.iloc[val_index].index, 'fold'] = fold_number
def _run():
    def loss_fn(outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
        model.train()
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(
                ids=ids,
                mask=mask)
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
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask
            )

            targets_np = targets.cpu().detach().numpy().tolist()
            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_targets.extend(targets_np)
            fin_outputs.extend(outputs_np)    

        return fin_outputs, fin_targets

    
    MAX_LEN = 192
    TRAIN_BATCH_SIZE = 1
    VALID_BATCH_SIZE = 1
    EPOCHS = 1

    FOLD = 0
    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("../input/xlm-roberta-base", do_lower_case=True)

    train_dataset = BERTDatasetTraining(
        premise=df_train[df_train['fold']!= FOLD].premise.values,
        hypothesis = df_train[df_train['fold']!= FOLD].hypothesis.values,
        targets=df_train[df_train['fold']!= FOLD].label.values,
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
        premise=df_train[df_train['fold']== FOLD].premise.values,
        hypothesis = df_train[df_train['fold']== FOLD].hypothesis.values,
        targets=df_train[df_train['fold']== FOLD].label.values,
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
        batch_size=VALID_BATCH_SIZE,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=4
    )

    device = xm.xla_device()
    model = mx.to(device)
#     chkp = torch.load("model.bin")
#     model.load_state_dict(chkp)
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
    best_auc = 0
    for epoch in range(EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)

        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        if auc> best_auc:
          best_auc = auc
          xm.save(model.state_dict(), "model.bin")

        xm.master_print(f'-------------\nEpochs: {epoch} \t AUC = {auc} \t best_AUC: {best_auc}\n------------')
# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run()

if train_enable:
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
MAX_LEN = 192
TEST_BATCH_SIZE = 4

class BERTDatasetTest:
    def __init__(self,premise, hypothesis, tokenizer, max_length):
        self.premise = premise
        self.hypothesis = hypothesis
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.premise)

    def __getitem__(self, item):
        hypothesis = str(self.hypothesis[item])
        premise = str(self.premise[item])

        hypothesis = " ".join(hypothesis.split())
        premise = " ".join(premise.split())
        hypothesis_inputs = self.tokenizer.encode_plus(
            hypothesis,
            None,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
            
        )
        premise_inputs = self.tokenizer.encode_plus(
            premise,
            None,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
            
        )
        ids = [0] + premise_inputs["input_ids"] + [2,2] + hypothesis_inputs['input_ids'] + [2]
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
        else:
            ids = ids[:self.max_length-1]+ [2]
            
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(masks, dtype=torch.long)
        }
    
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained("../input/xlm-roberta-base", do_lower_case=True)
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

mx = BERTBaseUncased()
device = 'xla:0'
model = mx.to(device)
checkpoint = ["../input/mberttrain10epochs/model-fold2.bin", 
              "../input/mberttrain10epochs/model-fold1.bin",
             "../input/mberttrain10epochs/model.bin"]
apt = []
for fold in checkpoint:
    chkp = torch.load(fold)
    mx.load_state_dict(chkp)
    mx.eval()
    fin_outputs = []
    for bi, d in enumerate(test_data_loader):
            ids = d["ids"]
            mask = d["mask"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            with torch.no_grad():
                outputs = model(
                        ids=ids,
                        mask=mask
                    )
            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_outputs.extend(outputs_np)
    apt.append(fin_outputs)

fin_concat = (apt[0]+ apt[2] + apt[3])/3
preds = np.argmax(fin_concat, axis=1)
# dic = {}
# oft = open('/kaggle/input/profanity/Profanity.txt', "r", encoding='utf8')
# for l in oft:
#     ele = l.strip().lower().split(':')
#     dic[ele[0]] = ele[1]
# oft.close()

# les, lit, ltr, lfr, lru, lpt = 1.2, 1.1, 1.3, 1.2, 1.2, 1.3
# of['content'] = test['content']
# of['tran'] = test['translated']
# of['lang'] = test['lang']
# out = []
# enpros = dic['en'].split(',')

# for _, row in of.iterrows():
#     if(row['lang']=='es'):
#         lmd = les
#     elif(row['lang']=='it'):
#         lmd = lit
#     elif(row['lang']=='tr'):
#         lmd = ltr
#     elif(row['lang']=='fr'):
#         lmd = lfr
#     elif(row['lang']=='ru'):
#         lmd = lru
#     else:
#         lmd = lpt

#     item = [row['id'], row['toxic']]
#     if(item[1]<0.5):
#         for w in enpros:
#             if(str(row['tran']).lower().find(w)>=0):
#                 item[1] *= 1.2
#                 break

#         ws = dic[row['lang']].split(',')
#         for w in ws:
#             if(str(row['content']).lower().find(w)>=0):
#                 item[1] *= lmd
#                 break
#     out.append(item)

# of = pd.DataFrame(out, columns=['id', 'toxic'])
# score1 = roc_auc_score(mybest.toxic.round().astype(int), of.toxic.values)
# score2 = roc_auc_score(of.toxic.round().astype(int), mybest.toxic.values)
# print('%2.4f\t%2.4f'%(100*score1, 100*score2))
df_submit = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
df_submit['prediction'] = preds
df_submit.head()
df_submit.to_csv("submission.csv", index=False)
