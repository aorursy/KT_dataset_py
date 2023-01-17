!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

!export XLA_USE_BF16=1
!pip install -q colored

!pip install -q transformers
import os

import gc

import re



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

from torch.optim.lr_scheduler import ReduceLROnPlateau



from torch.multiprocessing import Pipe, Process

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler



from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from transformers import RobertaModel, RobertaTokenizer



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp



from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences as pad
EPOCHS = 20

SPLIT = 0.8

MAXLEN = 48

DROP_RATE = 0.3

np.random.seed(42)



OUTPUT_UNITS = 3

BATCH_SIZE = 384

LR = (4e-5, 1e-2)

ROBERTA_UNITS = 768

VAL_BATCH_SIZE = 384

MODEL_SAVE_PATH = 'sentiment_model.pt'
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test_df.head()
train_df.head()
class TweetDataset(Dataset):

    def __init__(self, data, tokenizer):

        self.data = data

        self.text = data.text

        self.tokenizer = tokenizer

        self.sentiment = data.sentiment

        self.sentiment_dict = {"positive": 0, "neutral": 1, "negative": 2}



    def __len__(self):

        return len(self.data)



    def __getitem__(self, i):

        start, finish = 0, 2

        pg, tg = 'post', 'post'

        tweet = str(self.text[i]).strip()

        tweet_ids = self.tokenizer.encode(tweet)



        attention_mask_idx = len(tweet_ids) - 1

        if start not in tweet_ids: tweet_ids = start + tweet_ids

        tweet_ids = pad([tweet_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)



        attention_mask = np.zeros(MAXLEN)

        attention_mask[1:attention_mask_idx] = 1

        attention_mask = attention_mask.reshape((1, -1))

        if finish not in tweet_ids: tweet_ids[-1], attention_mask[-1] = finish, start

            

        sentiment = [self.sentiment_dict[self.sentiment[i]]]

        sentiment = torch.FloatTensor(to_categorical(sentiment, num_classes=3))

        return sentiment, torch.LongTensor(tweet_ids), torch.LongTensor(attention_mask)
class Roberta(nn.Module):

    def __init__(self):

        super(Roberta, self).__init__()

        self.softmax = nn.Softmax(dim=1)

        self.drop = nn.Dropout(DROP_RATE)

        self.roberta = RobertaModel.from_pretrained(model)

        self.dense = nn.Linear(ROBERTA_UNITS, OUTPUT_UNITS)

        

    def forward(self, inp, att):

        inp = inp.view(-1, MAXLEN)

        _, self.feat = self.roberta(inp, att)

        return self.softmax(self.dense(self.drop(self.feat)))
model = 'roberta-base'

tokenizer = RobertaTokenizer.from_pretrained(model)
def cel(inp, target):

    _, labels = target.max(dim=1)

    return nn.CrossEntropyLoss()(inp, labels)*len(inp)



def accuracy(inp, target):

    inp_ind = inp.max(axis=1).indices

    target_ind = target.max(axis=1).indices

    return (inp_ind == target_ind).float().sum(axis=0)
m = Roberta(); print(m)
del m; gc.collect()
def print_metric(data, batch, epoch, start, end, metric, typ):

    t = typ, metric, "%s", data, "%s"

    if typ == "Train": pre = "BATCH %s" + str(batch-1) + "%s  "

    if typ == "Val": pre = "\nEPOCH %s" + str(epoch+1) + "%s  "

    time = np.round(end - start, 1); time = "Time: %s{}%s s".format(time)

    fonts = [(fg(211), attr('reset')), (fg(212), attr('reset')), (fg(213), attr('reset'))]

    xm.master_print(pre % fonts[0] + "{} {}: {}{}{}".format(*t) % fonts[1] + "  " + time % fonts[2])
global val_losses; global train_losses

global val_accuracies; global train_accuracies



def train_fn(train_df):

    train_df = shuffle(train_df)

    train_df = train_df.reset_index(drop=True)



    split = np.int32(SPLIT*len(train_df))

    val_df, train_df = train_df[split:], train_df[:split]



    val_df = val_df.reset_index(drop=True)

    val_dataset = TweetDataset(val_df, tokenizer)

    val_sampler = DistributedSampler(val_dataset, num_replicas=8,

                                     rank=xm.get_ordinal(), shuffle=True)

    

    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE,

                            sampler=val_sampler, num_workers=0, drop_last=True)



    train_df = train_df.reset_index(drop=True)

    train_dataset = TweetDataset(train_df, tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=8,

                                       rank=xm.get_ordinal(), shuffle=True)



    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,

                              sampler=train_sampler, num_workers=0, drop_last=True)



    device = xm.xla_device()

    network = Roberta().to(device)

    optimizer = Adam([{'params': network.dense.parameters(), 'lr': LR[1]},

                      {'params': network.roberta.parameters(), 'lr': LR[0]}])



    val_losses, val_accuracies = [], []

    train_losses, train_accuracies = [], []

    

    start = time.time()

    xm.master_print("STARTING TRAINING ...\n")



    for epoch in range(EPOCHS):



        batch = 1

        network.train()

        fonts = (fg(48), attr('reset'))

        xm.master_print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)



        val_parallel = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)

        train_parallel = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

        

        for train_batch in train_parallel:

            train_targ, train_in, train_att = train_batch

            

            network = network.to(device)

            train_in = train_in.to(device)

            train_att = train_att.to(device)

            train_targ = train_targ.to(device)



            train_preds = network.forward(train_in, train_att)

            train_loss = cel(train_preds, train_targ.squeeze(dim=1))/len(train_in)

            train_accuracy = accuracy(train_preds, train_targ.squeeze(dim=1))/len(train_in)



            optimizer.zero_grad()

            train_loss.backward()

            xm.optimizer_step(optimizer)

            

            end = time.time()

            batch = batch + 1

            acc = np.round(train_accuracy.item(), 3)

            print_metric(acc, batch, None, start, end, metric="acc", typ="Train")



        val_loss, val_accuracy, val_points = 0, 0, 0



        network.eval()

        with torch.no_grad():

            for val_batch in val_parallel:

                targ, val_in, val_att = val_batch



                targ = targ.to(device)

                val_in = val_in.to(device)

                val_att = val_att.to(device)

                network = network.to(device)

            

                val_points += len(targ)

                pred = network.forward(val_in, val_att)

                val_loss += cel(pred, targ.squeeze(dim=1)).item()

                val_accuracy += accuracy(pred, targ.squeeze(dim=1)).item()

        

        end = time.time()

        val_loss /= val_points

        val_accuracy /= val_points

        acc = xm.mesh_reduce('acc', val_accuracy, lambda x: sum(x)/len(x))

        print_metric(np.round(acc, 3), None, epoch, start, end, metric="acc", typ="Val")

    

        xm.master_print("")

        val_losses.append(val_loss); train_losses.append(train_loss.item())

        val_accuracies.append(val_accuracy); train_accuracies.append(train_accuracy.item())



    xm.master_print("ENDING TRAINING ...")

    xm.save(network.state_dict(), MODEL_SAVE_PATH); del network; gc.collect()



    metric_names = ['val_loss_', 'train_loss_', 'val_acc_', 'train_acc_']

    metric_lists = [val_losses, train_losses, val_accuracies, train_accuracies]

    

    for i, metric_list in enumerate(metric_lists):

        for j, metric_value in enumerate(metric_list):

            torch.save(metric_value, metric_names[i] + str(j) + '.pt')
FLAGS = {}

def _mp_fn(rank, flags): train_fn(train_df)

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
val_losses = [torch.load('val_loss_{}.pt'.format(i)) for i in range(EPOCHS)]

train_losses = [torch.load('train_loss_{}.pt'.format(i)) for i in range(EPOCHS)]

val_accuracies = [torch.load('val_acc_{}.pt'.format(i)) for i in range(EPOCHS)]

train_accuracies = [torch.load('train_acc_{}.pt'.format(i)) for i in range(EPOCHS)]
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1, len(val_losses)+1),

                         y=val_losses, mode="lines+markers", name="val",

                         marker=dict(color="hotpink", line=dict(width=.5,

                                                                color='rgb(0, 0, 0)'))))



fig.add_trace(go.Scatter(x=np.arange(1, len(train_losses)+1),

                         y=train_losses, mode="lines+markers", name="train",

                         marker=dict(color="mediumorchid", line=dict(width=.5,

                                                                     color='rgb(0, 0, 0)'))))



fig.update_layout(xaxis_title="Epochs", yaxis_title="Cross Entropy",

                  title_text="Cross Entropy vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0")



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=np.arange(1, len(val_accuracies)+1),

                         y=val_accuracies, mode="lines+markers", name="val",

                         marker=dict(color="hotpink", line=dict(width=.5,

                                                                color='rgb(0, 0, 0)'))))



fig.add_trace(go.Scatter(x=np.arange(1, len(train_accuracies)+1),

                         y=train_accuracies, mode="lines+markers", name="train",

                         marker=dict(color="mediumorchid", line=dict(width=.5,

                                                                     color='rgb(0, 0, 0)'))))



fig.update_layout(xaxis_title="Epochs", yaxis_title="Accuracy",

                  title_text="Accuracy vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0")



fig.show()
network = Roberta()

network.load_state_dict(torch.load('sentiment_model.pt'))
device = xm.xla_device()

network = network.to(device)



def predict_sentiment(tweet):

    pg, tg = 'post', 'post'

    tweet_ids = tokenizer.encode(tweet.strip())

    sent = {0: 'positive', 1: 'neutral', 2: 'negative'}



    att_mask_idx = len(tweet_ids) - 1

    if 0 not in tweet_ids: tweet_ids = 0 + tweet_ids

    tweet_ids = pad([tweet_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)



    att_mask = np.zeros(MAXLEN)

    att_mask[1:att_mask_idx] = 1

    att_mask = att_mask.reshape((1, -1))

    if 2 not in tweet_ids: tweet_ids[-1], att_mask[-1] = 2, 0

    tweet_ids, att_mask = torch.LongTensor(tweet_ids), torch.LongTensor(att_mask)

    return sent[np.argmax(network.forward(tweet_ids.to(device), att_mask.to(device)).detach().cpu().numpy())]
predict_sentiment("It does not look good now ...")
predict_sentiment("I want to know more about your product.")
predict_sentiment("I have done something good today and so should you :D")