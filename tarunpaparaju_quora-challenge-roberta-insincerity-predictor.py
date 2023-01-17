!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!export XLA_USE_BF16=1

!pip install -q colored
import os

import gc

import time

import colored

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from colored import fg, bg, attr



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import torch

import torch.nn as nn

from torch.optim import Adam

from torch import LongTensor, FloatTensor, DoubleTensor

from torch.utils.data import Dataset, DataLoader, sampler

from torch.utils.data.distributed import DistributedSampler



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp



from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from transformers import RobertaModel, RobertaTokenizer



from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences as pad
EPOCHS = 5

MAXLEN = 64

SPLIT = 0.8

DROP_RATE = 0.3

LR = (4e-5, 1e-2)

BATCH_SIZE = 256

VAL_BATCH_SIZE = 8192

MODEL_SAVE_PATH = 'insincerity_model.pt'
np.random.seed(42)

torch.manual_seed(42)
test_df = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

train_df = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df.head()
train_df.head()
class QuoraDataset(Dataset):

    def __init__(self, data, tokenizer):

        self.text = data.question_text

        self.data, self.tokenizer = data, tokenizer

        self.target = data.target if "target" in data.columns else [-1]*len(data)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, i):

        pg, tg = 'post', 'post'

        target = [self.target[i]]

        question = str(self.text[i])

        quest_ids = self.tokenizer.encode(question.strip())



        attention_mask_idx = len(quest_ids) - 1

        if 0 not in quest_ids: quest_ids = 0 + quest_ids

        quest_ids = pad([quest_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)



        attention_mask = np.zeros(MAXLEN)

        attention_mask[1:attention_mask_idx] = 1

        attention_mask = attention_mask.reshape((1, -1))

        if 2 not in quest_ids: quest_ids[-1], attention_mask[-1] = 2, 0

        return FloatTensor(target), LongTensor(quest_ids), LongTensor(attention_mask)
model = 'roberta-base'

tokenizer = RobertaTokenizer.from_pretrained(model)
class Roberta(nn.Module):

    def __init__(self):

        super(Roberta, self).__init__()

        self.dropout = nn.Dropout(DROP_RATE)

        self.dense_output = nn.Linear(768, 1)

        self.roberta = RobertaModel.from_pretrained(model)



    def forward(self, inp, att):

        inp = inp.view(-1, MAXLEN)

        _, self.feat = self.roberta(inp, att)

        return self.dense_output(self.dropout(self.feat))
m = Roberta()
print(m); del m; gc.collect()
def bce(y_pred, y_true):

    return nn.BCEWithLogitsLoss()(y_pred, y_true)*len(y_pred)



def f1_score(y_pred, y_true):

    y_true = y_true.squeeze()

    y_pred = torch.round(nn.Sigmoid()(y_pred)).squeeze()

    

    tp = (y_true * y_pred).sum().to(torch.float32)

    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)

    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)

    

    epsilon = 1e-7

    recall = tp / (tp + fn + epsilon)

    precision = tp / (tp + fp + epsilon)

    return 2*(precision*recall) / (precision + recall + epsilon)
def print_metric(data, batch, epoch, start, end, metric, typ):

    t = typ, metric, "%s", data, "%s"

    if typ == "Train": pre = "BATCH %s" + str(batch-1) + "%s  "

    if typ == "Val": pre = "\nEPOCH %s" + str(epoch+1) + "%s  "

    time = np.round(end - start, 1); time = "Time: %s{}%s s".format(time)

    fonts = [(fg(211), attr('reset')), (fg(212), attr('reset')), (fg(213), attr('reset'))]

    xm.master_print(pre % fonts[0] + "{} {}: {}{}{}".format(*t) % fonts[1] + "  " + time % fonts[2])
global val_f1s; global train_f1s

global val_losses; global train_losses



def train_fn(df):

    split = np.int32(SPLIT*len(df))

    val_df, train_df = df[split:], df[:split]



    val_df = val_df.reset_index(drop=True)

    val_dataset = QuoraDataset(val_df, tokenizer)

    val_sampler = DistributedSampler(val_dataset, num_replicas=8,

                                     rank=xm.get_ordinal(), shuffle=True)

    

    val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_BATCH_SIZE,

                            sampler=val_sampler, num_workers=0, drop_last=True)



    train_df = train_df.reset_index(drop=True)

    train_dataset = QuoraDataset(train_df, tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=8,

                                       rank=xm.get_ordinal(), shuffle=True)



    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,

                              sampler=train_sampler, num_workers=0, drop_last=True)



    device = xm.xla_device()

    network = Roberta().to(device)

    optimizer = Adam([{'params': network.roberta.parameters(), 'lr': LR[0]},

                      {'params': network.dense_output.parameters(), 'lr': LR[1]}])



    val_losses, val_f1s = [], []

    train_losses, train_f1s = [], []

    

    start = time.time()

    xm.master_print("STARTING TRAINING ...\n")



    for epoch in range(EPOCHS):

        fonts = (fg(48), attr('reset'))

        xm.master_print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)



        val_parallel = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)

        train_parallel = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

        

        batch = 1

        network.train()

        for train_batch in train_parallel:

            train_targ, train_in, train_att = train_batch



            train_in = train_in.to(device)

            train_att = train_att.to(device)

            train_targ = train_targ.to(device)

            train_preds = network.forward(train_in, train_att)



            train_loss = bce(train_preds, train_targ)/len(train_preds)

            train_f1 = f1_score(train_preds, train_targ.squeeze(dim=1))



            optimizer.zero_grad()

            train_loss.backward()

            xm.optimizer_step(optimizer)

            

            end = time.time()

            batch = batch + 1

            is_print = batch % 10 == 1

            f1 = np.round(train_f1.item(), 3)

            if is_print: print_metric(f1, batch, None, start, end, "F1", "Train")



        val_loss, val_f1, val_points = 0, 0, 0



        network.eval()

        with torch.no_grad():

            for val_batch in val_parallel:

                targ, val_in, val_att = val_batch

                

                targ = targ.to(device)

                val_in = val_in.to(device)

                val_att = val_att.to(device)

                pred = network.forward(val_in, val_att)



                val_points += len(targ)

                val_loss += bce(pred, targ).item()

                val_f1 += f1_score(pred, targ.squeeze(dim=1)).item()*len(pred)

        

        end = time.time()

        val_f1 /= val_points

        val_loss /= val_points

        f1 = xm.mesh_reduce('f1', val_f1, lambda x: sum(x)/len(x))

        loss = xm.mesh_reduce('loss', val_loss, lambda x: sum(x)/len(x))

        print_metric(np.round(f1, 3), None, epoch, start, end, "F1", "Val")

    

        xm.master_print("")

        val_f1s.append(f1); train_f1s.append(train_f1.item())

        val_losses.append(loss); train_losses.append(train_loss.item())



    xm.master_print("ENDING TRAINING ...")

    xm.save(network.state_dict(), MODEL_SAVE_PATH); del network; gc.collect()

    

    metric_lists = [val_losses, train_losses, val_f1s, train_f1s]

    metric_names = ['val_loss_', 'train_loss_', 'val_f1_', 'train_f1_']

    

    for i, metric_list in enumerate(metric_lists):

        for j, metric_value in enumerate(metric_list):

            torch.save(metric_value, metric_names[i] + str(j) + '.pt')
FLAGS = {}

train_df = shuffle(train_df)

train_df = train_df.reset_index(drop=True)



def _mp_fn(rank, flags): train_fn(df=train_df)

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
val_f1s = [0] + [torch.load('val_f1_{}.pt'.format(i)) for i in range(EPOCHS)]

train_f1s = [0] + [torch.load('train_f1_{}.pt'.format(i)) for i in range(EPOCHS)]

val_losses = [0.25] + [torch.load('val_loss_{}.pt'.format(i)) for i in range(EPOCHS)]

train_losses = [0.25] + [torch.load('train_loss_{}.pt'.format(i)) for i in range(EPOCHS)]
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1, len(val_losses)+1),

                         y=val_losses, mode="lines+markers", name="val",

                         marker=dict(color="indianred", line=dict(width=.5,

                                                                  color='rgb(0, 0, 0)'))))



fig.add_trace(go.Scatter(x=np.arange(1, len(train_losses)+1),

                         y=train_losses, mode="lines+markers", name="train",

                         marker=dict(color="darkorange", line=dict(width=.5,

                                                                   color='rgb(0, 0, 0)'))))



fig.update_layout(xaxis_title="Epochs", yaxis_title="Binary Cross Entropy",

                  title_text="Binary Cross Entropy vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0")



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1, len(val_f1s)+1),

                         y=val_f1s, mode="lines+markers", name="val",

                         marker=dict(color="indianred", line=dict(width=.5,

                                                                  color='rgb(0, 0, 0)'))))



fig.add_trace(go.Scatter(x=np.arange(1, len(train_f1s)+1),

                         y=train_f1s, mode="lines+markers", name="train",

                         marker=dict(color="darkorange", line=dict(width=.5,

                                                                   color='rgb(0, 0, 0)'))))



fig.update_layout(xaxis_title="Epochs", yaxis_title="F1 Score",

                  title_text="F1 Score vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0")



fig.show()
network = Roberta()

network.load_state_dict(torch.load('insincerity_model.pt'))
device = xm.xla_device()

network = network.to(device); network = network.eval()



def predict_insincerity(question):

    pg, tg = 'post', 'post'

    ins = {0: 'sincere', 1: 'insincere'}

    quest_ids = tokenizer.encode(question.strip())



    attention_mask_idx = len(quest_ids) - 1

    if 0 not in quest_ids: quest_ids = 0 + quest_ids

    quest_ids = pad([quest_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)



    att_mask = np.zeros(MAXLEN)

    att_mask[1:attention_mask_idx] = 1

    att_mask = att_mask.reshape((1, -1))

    if 2 not in quest_ids: quest_ids[-1], attention_mask[-1] = 2, 0

    quest_ids, att_mask = torch.LongTensor(quest_ids), torch.LongTensor(att_mask)

    

    output = network.forward(quest_ids.to(device), att_mask.to(device))

    return ins[int(np.round(nn.Sigmoid()(output.detach().cpu()).item()))]
predict_insincerity("How can I train roBERTa base on TPUs?")
predict_insincerity("Why is that stupid man the biggest dictator in the world?")
def sigmoid(x):

    return 1/(1 + np.exp(-x))
network.eval()

test_preds = []



test_dataset = QuoraDataset(test_df, tokenizer)

test_loader = tqdm(DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE))



with torch.no_grad():

    for batch in test_loader:

        _, test_in, test_att = batch

        test_in = test_in.to(device)

        test_att = test_att.to(device)

        test_pred = network.forward(test_in, test_att)

        test_preds.extend(test_pred.squeeze().detach().cpu().numpy())



test_preds = np.int32(np.round(sigmoid(np.array(test_preds))))
path = '../input/quora-insincere-questions-classification/'

sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission.prediction = test_preds
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)