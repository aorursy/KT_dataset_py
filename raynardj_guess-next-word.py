!pip install git+https://github.com/raynardj/forge
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sqlalchemy import create_engine as ce

from sqlalchemy import inspect

from pathlib import Path
DATA= Path("/kaggle/input/classic-english-literature-corpus/books.db")



engine = ce("sqlite:///"+str(DATA))

inspector = inspect(engine)



print(inspector.get_table_names())
books_df = pd.read_sql("books", con = engine)

author_df = pd.read_sql("authors", con = engine)

book_file_df = pd.read_sql("book_file", con = engine)

text_file_df = pd.read_sql("text_files", con = engine)
def searchAuthor(kw):

    return author_df[author_df.author.str.contains(kw)]



def searchBookByAuthor(kw):

    author_result = list(searchAuthor(kw).index)

    return books_df[books_df.author_id.isin(author_result)]

AUTHOR = "Dickens"

searchBookByAuthor(AUTHOR).head()
def getAllTextByAuthor(kw):

    """

    kw: str, key words

    search the authors by key word, if you search John, it would relate to all the author whose name contains John

    return: str, all the textual content in 1 long lower case string

    """

    file_ids = book_file_df[book_file_df.book_id.isin(searchBookByAuthor(kw).book_id)].file_id.unique()

    all_text = "\n".join(list(text_file_df[text_file_df.index.isin(file_ids)].text))

    return all_text.replace("\n"," ").replace("   "," ").replace("  "," ").replace("\t"," ").lower()



all_text = getAllTextByAuthor(AUTHOR)
print(all_text[:500])
from nltk import TweetTokenizer

tk = TweetTokenizer()



txt = tk.tokenize(all_text)
order = 10

datalist = []

for i in range(order):

    txt_ = txt[i:]

    datalist += list("<tok>".join(txt_[bsidx:bsidx+order]) for bsidx in range(0,len(txt_),order))
CLASS_BALANCE = False
txtdf = pd.DataFrame({"txt":datalist}).reset_index().rename(columns = {"index":"id"})

txtdf["y"] = txtdf.txt.apply(lambda x:x.split("<tok>")[-1])

if CLASS_BALANCE:

    txtdf = pd.merge(txtdf,pd.DataFrame({"ycount":txtdf.groupby("y")["id"].apply(lambda x:len(x))}), on = "y", how="left")

    txtdf["rand_num"] = np.random.rand(len(txtdf))/(txtdf.ycount.apply(lambda x:0.01 if x<500 else x).values.astype(np.float32))

    

    txtdf = txtdf[txtdf.rand_num > 5e-4]
txtdf
from torch.utils.data.dataset import Dataset

from torch.utils.data.dataloader import DataLoader

from collections import Counter

import math



from forgebox.ftorch.prepro import Seq_Dataset,fuse,test_DS
seq = Seq_Dataset("text", # sequence name

                  txtdf.txt, # pandas data series

                  bs = 64, # batch size

                  vocab_size = 10000,

                  seq_len=order, #sequence length

                  sep_tok="<tok>", fixlen=True,build_vocab = True, 

                  vocab_path="vocab_tok.json")
seq.vocab.tail()
import os

import torch

from torch import nn



def getidx(x):

    try: return seq.char2idx[x]

    except: return -1

    

txtdf["missing"] = txtdf.y.apply(getidx) 

txtdf = txtdf[txtdf.missing>-1]

print(len(txtdf))

seq = Seq_Dataset("text", # sequence name

                  txtdf.txt, # pandas data series

                  bs = 64, # batch size

                  vocab_size = 10000,

                  seq_len=order, #sequence length

                  sep_tok="<tok>", fixlen=True,build_vocab = False, 

                  vocab_path="vocab_tok.json")


dt = test_DS(seq)()[0]

dt
CUDA = torch.cuda.is_available()

print(CUDA)
class Writer(nn.Module):

    def __init__(self, hs, num_layers =1, scale=1, vs = len(seq.vocab),dropout = 0.3):

        super().__init__()

        self.nl = num_layers

        self.encoder = nn.Embedding(vs,hs)

        self.drop = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_size=hs, hidden_size=hs*scale, num_layers=self.nl, batch_first=True, dropout =  dropout)

        self.bn = nn.BatchNorm1d(hs*scale)

        self.decoder = nn.Linear(hs*scale,vs)

        self.init_hn = torch.zeros(self.nl,1,hs*scale)

        self.init_cn = torch.zeros(self.nl,1,hs*scale)

        

    def hn_cn(self,x):

        """

        Initialize cell state and hidden state

        """

        bs = x.size(0)

        return self.init_hn.repeat([1,bs,1]),self.init_cn.repeat([1,bs,1])

    

    def init_weights(self):

        initrange = 0.1

        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()

        self.decoder.weight.data.uniform_(-initrange, initrange)

        

    def forward(self, x,hn,cn):

        bs =x.size(0)

        x = self.encoder(x)

        x,(hn, cn) = self.rnn(x,(hn, cn))

        x = hn.view(bs,-1)

        x = self.drop(x)

        x = self.bn(x)

        x = self.decoder(x)

        return x,(hn, cn)
MODEL = "lm_model.npy"
from forgebox.ftorch.train import Trainer

from forgebox.ftorch.callbacks import stat
model = Writer(200,scale = 1,dropout = 0.1)



if CUDA: model.cuda()

opt = torch.optim.Adam(model.parameters())



# Loss function, it's a classification problemset => cross entropy

crit = nn.CrossEntropyLoss()
model.train()
CLIP = 0.25

import random

t = Trainer(seq,callbacks=[stat],batch_size=1, shuffle=True)

@t.step_train

def action(batch):

    data = batch.data[0]

    if CUDA: data = data.cuda()

    

    opt.zero_grad()

    x = data[:,:-1]

    y = data[:,-1:]

        

    hn, cn  = model.hn_cn(data)

    if CUDA:

        hn = hn.cuda()

        cn = cn.cuda()

        

    y_, (hn, cn) = model(x,hn,cn)



    loss = crit(y_,y.squeeze(-1))

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)

    opt.step()

    

    if batch.i ==0:

        torch.save(model.state_dict(), MODEL)

        

    return {"loss":loss.item(), "acc":(torch.max(y_,keepdim=True, dim=-1)[1]==y).float().mean().item()}
t.train(3)