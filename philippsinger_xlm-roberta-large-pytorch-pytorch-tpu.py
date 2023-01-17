import tensorflow as tf

try:

   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

except ValueError:

   tpu = None

if tpu:

   tf.config.experimental_connect_to_cluster(tpu)

   tf.tpu.experimental.initialize_tpu_system(tpu)

   strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

   strategy = tf.distribute.get_strategy()
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly  --apt-packages libomp5 libopenblas-dev

import os



os.environ['XLA_USE_BF16'] = "1"

os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'



import torch

import pandas as pd

from scipy import stats

import numpy as np



import gc



from tqdm import tqdm

from collections import OrderedDict, namedtuple

import torch.nn as nn

from torch.optim import lr_scheduler

import joblib

from joblib import Parallel, delayed



import torch_xla.utils.serialization as xser



import time



import logging

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig, get_cosine_schedule_with_warmup

import sys

from sklearn import metrics, model_selection

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from tqdm.notebook import tqdm



from random import shuffle

import random



import re



import warnings

import torch_xla

import torch_xla.debug.metrics as met

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils

import warnings

warnings.filterwarnings("ignore")
PATH = ""



MAX_LEN = 224
df_train = pd.read_csv("../input/jigsaw-public-baseline-train-data/train_data.csv", usecols=["comment_text", "toxic", "lang"])



df_valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv', usecols=["comment_text", "toxic", "lang"])



df_test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv').rename(columns={"content": "comment_text"})
df_train.head()
# create (balanced) samples outside the training routine to save memory



labels = np.char.add(df_train.toxic.values.astype(str), df_train.lang.values)

df_train["label"] = labels



min_size = df_train.groupby("label").size().min()



print(min_size)



upsample = 1

samples = []

for i in range(3):

    print(i)

    sample = []

    for l in df_train.label.unique():

        if l[0] == "1":

            x = df_train[df_train["label"]==l].sample(min_size, replace=False, random_state=i)

            sample.append(x)

            sample.append(df_train[df_train["label"]==f"0{l[1:]}"].sample(min_size*upsample, replace=False, random_state=i))

    sample = pd.concat(sample, axis=0).sample(frac=1)

    del sample["label"]

    DATA_LENGTH = len(sample)

    samples.append(sample)

    del sample
del df_train

df_train = samples
gc.collect()

!free -h
class CustomRoberta(nn.Module):

    def __init__(self):

        super(CustomRoberta, self).__init__()

        self.num_labels = 2

        self.roberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-large", output_hidden_states=False, num_labels=1)

        self.dropout = nn.Dropout(p=0.2)

        self.ln = nn.LayerNorm(1024)

        self.classifier = nn.Linear(1024, self.num_labels)



    def forward(self,

                input_ids=None,

                attention_mask=None,

                position_ids=None,

                head_mask=None,

                inputs_embeds=None):



        o1, o2 = self.roberta(input_ids,

                               attention_mask=attention_mask,

                               position_ids=position_ids,

                               head_mask=head_mask,

                               inputs_embeds=inputs_embeds)

        

        x1 = torch.mean(o1, 1)

        

        x = x1

        

        x = self.ln(x)

        x = self.dropout(x)



        logits = self.classifier(x)       

        

        return logits
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')



# use model wrapper for reducing memory usage across TPU cores

mx = xmp.MpModelWrapper(CustomRoberta())
class BERTDataset:

    def __init__(self, df=None):

        self.df = df



    def __len__(self):

        return len(self.df)

    

    def get_tokens(self, text):

        encoded = tokenizer.encode_plus(

            text, 

            add_special_tokens=True, 

            max_length=MAX_LEN, 

            pad_to_max_length=True

        )

        return encoded['input_ids'], encoded['attention_mask']

    

    def get_labels(self):

        return LabelEncoder().fit_transform(np.char.add(self.df.toxic.values.astype(str), self.df.lang.values).reshape(-1,1)).astype(np.int16)



    def __getitem__(self, item):

        

        text = self.df.iloc[item]["comment_text"]

        lang = self.df.iloc[item]["lang"]

                

        encoded = self.get_tokens(text)

        

        targets = np.zeros(2)

        

        if "toxic" in self.df.columns:

            targets[self.df.iloc[item]["toxic"]] = 1

        

        return {

            'ids': torch.tensor(encoded[0]),

            'mask': torch.tensor(encoded[1]),

            'targets': targets,

            'index': item

        }
#train_dataset = BERTDataset(df_train)

valid_dataset = BERTDataset(df_valid)

test_dataset = BERTDataset(df_test)
gc.collect()

!free -h
class RocAucMeter(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.y_true = np.array([0,1])

        self.y_pred = np.array([0.5,0.5])

        self.score = 0



    def update(self, y_true, y_pred):

        y_true = y_true.cpu().numpy().argmax(axis=1)

        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]

        self.y_true = np.hstack((self.y_true, y_true))

        self.y_pred = np.hstack((self.y_pred, y_pred))

        self.score = metrics.roc_auc_score(self.y_true, self.y_pred, labels=np.array([0, 1]))

    

    @property

    def avg(self):

        return self.score



class AverageMeter(object):

    """Computes and stores the average and current value"""

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



def train_loop_fn(data_loader, model, optimizer, device, scheduler=None, epoch=None):

        

    model.train()



    losses = AverageMeter()

    auc = RocAucMeter()

    start_time = time.time()

    

    for bi, d in enumerate(data_loader):



        ids = d["ids"]

        mask = d["mask"]

        targets = d["targets"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)



        optimizer.zero_grad()

        

        outputs = model(

            input_ids=ids,

            attention_mask = mask,

        )



        

        #xm.master_print(f'{outputs.shape}')

        #xm.master_print(f'{targets.shape}')

        loss = loss_fn(outputs, targets)

        

        loss.backward()

        xm.optimizer_step(optimizer)

        

        loss = loss.detach().item()

        

        auc.update(targets, outputs)

        losses.update(loss, ids.size(0))

        

        if bi % 10 == 0:

            xm.master_print(f'bi={bi}, loss={losses.avg:<8.4f}, auc={auc.avg:<8.4f} {time.time()-start_time:<2.2f}')



        if scheduler is not None:

            scheduler.step()

        #break

        #break

#         if bi == 2:

#             break

        

    del loss

    del losses

    del outputs

    del ids

    del targets

    

    gc.collect()

        

    model.eval()



def eval_loop_fn(data_loader, model, device):

        

    #model.eval()

    fin_targets = []

    fin_outputs = []

    fin_index = []

    with torch.no_grad():

        for bi, d in enumerate(data_loader):



            if bi % 10 == 0:

                xm.master_print(f'EVAL bi={bi}')



            ids = d["ids"]

            mask = d["mask"]

            targets = d["targets"]

            index = d["index"]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)



            outputs = model(

                input_ids = ids,

                attention_mask = mask,

            )



            targets_np = targets.cpu().detach().numpy().argmax(axis=1).tolist()

            outputs_np = outputs.cpu().detach().numpy()[:,1].tolist()

            fin_targets.extend(targets_np)

            fin_outputs.extend(outputs_np)    

            fin_index.extend(index.tolist()) 



    return fin_outputs, fin_targets, fin_index

class SmoothLoss(nn.Module):

    def __init__(self):

       super(SmoothLoss, self).__init__()

    def forward(self, pred, target):

       pred = pred.log_softmax(dim=1)

       return torch.mean(torch.sum(-target * pred, dim=1))

        

def loss_fn(outputs, targets):

    return SmoothLoss()(outputs, targets)



TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 32



EPOCHS = 1



LR = 2e-5



def _run():

    

    gc.collect()

    

    xm.master_print('starting run')

    

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

        num_workers=0

    )

    

    test_sampler = torch.utils.data.distributed.DistributedSampler(

              test_dataset,

              num_replicas=xm.xrt_world_size(),

              rank=xm.get_ordinal(),

              shuffle=False)



    test_data_loader = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=VALID_BATCH_SIZE,

        sampler=test_sampler,

        drop_last=False,

        num_workers=0

    )

    

    gc.collect()

    

    device = xm.xla_device()

    model = mx.to(device)

    xm.master_print('done loading model')





    num_train_steps = int(len(df_train[0]) / TRAIN_BATCH_SIZE / xm.xrt_world_size())



    optimizer = AdamW([{'params': model.roberta.parameters(), 'lr': LR},

                    {'params': [param for name, param in model.named_parameters() if 'roberta' not in name], 'lr': 1e-3} ], lr=LR, weight_decay=0)



    scheduler = get_cosine_schedule_with_warmup(

        optimizer,

        num_warmup_steps = 0,

        num_training_steps = num_train_steps * EPOCHS

    )



    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')



    for epoch in range(EPOCHS):



        # loading dataset for epoch

        train_dataset = BERTDataset(df_train[epoch])

    

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

            shuffle=False

        )

        

        train_sampler.set_epoch(epoch)

        

        para_loader = pl.ParallelLoader(train_data_loader, [device])

        xm.master_print('parallel loader created... training now')

        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler, epoch=epoch)

        

        #del train_dataset

        #del train_sampler

        #del train_data_loader

        del para_loader

        gc.collect()

        

        # using xm functionality for memory-reduced model saving

        if epoch == EPOCHS-1:

            xm.master_print('saving model')

            xser.save(model.state_dict(), f"{PATH}model.bin", master_only=True)

            xm.master_print('model saved')

        

        para_loader = pl.ParallelLoader(valid_data_loader, [device])

        o, t, i = eval_loop_fn(para_loader.per_device_loader(device), model, device)

        auc = metrics.roc_auc_score(np.array(t), o)

        #del o,t,i

        gc.collect()

        

        del para_loader



        print(f'[xla:{xm.get_ordinal()}] AUC = {auc}')

        

        

        def reduce_fn(vals):

            return sum(vals) / len(vals)



        auc = xm.mesh_reduce('auc_reduce', auc, reduce_fn)

        xm.master_print(f'AUC AVG = {auc}')

        

        para_loader = pl.ParallelLoader(test_data_loader, [device])

        o, t, i = eval_loop_fn(para_loader.per_device_loader(device), model, device)

        

        del t

        gc.collect()

        

    return o, i

        
gc.collect()



# Start training processes

def _mp_fn(rank, flags):

    

    # not the cleanest way, but works

    # collect individual core outputs and save

    # can also do test inference outside training routine loading saved model

    test_preds, test_index = _run()

    np.save(f"test_preds_{rank}", test_preds)

    np.save(f"test_index_{rank}", test_index)

    return test_preds



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
# showcase for loading data and inference



TRAIN_BATCH_SIZE = 64



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=TRAIN_BATCH_SIZE,

    drop_last=False,

    num_workers=0,

    shuffle=False

)



device = xm.xla_device()

model = mx.to(device).eval()

model.load_state_dict(xser.load(f"{PATH}model.bin"))



fin_targets = []

test_preds = []

for bi, d in enumerate(valid_data_loader):



    if bi % 50 == 0:

        xm.master_print(f'EVAL bi={bi}')



    ids = d["ids"]

    mask = d["mask"]

    targets = d["targets"]

    index = d["index"]



    ids = ids.to(device, dtype=torch.long)

    mask = mask.to(device, dtype=torch.long)

    targets = targets.to(device, dtype=torch.float)



    outputs = model(

        input_ids = ids,

        attention_mask = mask,

    )  



    targets_np = targets.cpu().detach().numpy().tolist()

    outputs_np = outputs.cpu().detach().numpy().tolist()

    fin_targets.extend(targets_np)

    test_preds.extend(outputs_np)   



test_preds = np.array(test_preds)

auc = metrics.roc_auc_score(df_valid.toxic.values, test_preds[:,1])

print(auc)

np.save("oof", test_preds)



for lang in df_valid.lang.unique():

    print(lang)

    print(metrics.roc_auc_score(df_valid[df_valid.lang==lang].toxic.values, test_preds[:,1][df_valid.lang==lang]))

    
# load individual outputs

test_preds = np.zeros(len(df_test))

for i in range(8):

    test_preds[np.load(f"test_index_{i}.npy", allow_pickle=True).reshape(-1)] = np.load(f"test_preds_{i}.npy", allow_pickle=True).reshape(-1)
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

sub['toxic'] = test_preds
sub.to_csv('submission.csv', index=False)
sub.head(10)