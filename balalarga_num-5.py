# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import nltk

from tqdm import tqdm_notebook

from collections import Counter

from nltk.tokenize import TweetTokenizer

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.functional as F

import torch.optim as optimizer

from torch.utils.data import DataLoader, Dataset
seed = 5143

np.random.seed(seed)

torch.manual_seed(seed)
root_dir = "../input/aclimdb/aclImdb/"

test_dir = root_dir + "test/"

train_dir = root_dir + "train/"

print(os.listdir(test_dir))

print(os.listdir(train_dir))
vocab = None

vocab_max = 6000

vocab_min = 15

text_len  = 100

with open(root_dir + "imdb.vocab") as f:

    vocab = {w.lower():k for k, w in enumerate(f.read().split()) if k <= vocab_max+vocab_min and k > vocab_min}

print(len(vocab))
def check_text(path):

    avg_len = 0

    counter = 0

    for p in ("/pos/", "/neg/"):

        files = os.listdir(path + p)

        for c,i in tqdm_notebook(enumerate(files), total=len(files)):

                avg_len += len(open(path + p + i).read().split())

                counter += 1

    print(avg_len/counter)
#check_text(train_dir)

#check_text(test_dir)
def getdata(path, vocab, max_len):

    data   = []

    labels = []

    label_val = 1

    for p in ("/pos/", "/neg/"):

        files = os.listdir(path + p)

        for _,i in tqdm_notebook(enumerate(files), total=len(files)):

            labels.append(label_val)

            file = open(path + p + i)

            text = TweetTokenizer().tokenize(file.read().lower())

            tok_text = []

            for word in text:

                tok_word = vocab.get(word)

                if tok_word is not None:

                    tok_text.append(tok_word)

            if len(tok_text) > max_len:

                tok_text = tok_text[:max_len]

            elif len(tok_text) < max_len:

                tok_text += [0 for i in range(max_len-len(tok_text))]

            data.append(tok_text)

            file.close()

        label_val = 0

    return data, labels
train_x, train_y = getdata(train_dir, vocab, text_len)
print(train_x[1])
test_x, test_y = getdata(test_dir, vocab, text_len)
valid_x, pred_x, valid_y, pred_x = train_test_split(test_x, test_y, test_size=0.7)
class Textset(Dataset):

    def __init__(self, x, y):

        self.data = x

        self.labels = y

        

    def __getitem__(self, i):

        return torch.tensor(data=self.data[i]), torch.FloatTensor(data=[self.labels[i]])

    

    def __len__(self):

        return len(self.labels)
trainset = Textset(train_x, train_y)

trainloader = DataLoader(trainset, batch_size=30, shuffle=True)
validset = Textset(valid_x, valid_y)

validloader = DataLoader(validset, batch_size=30)
testset = Textset(test_x, test_y)

testloader = DataLoader(testset, batch_size=30)
class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, batch_size, hid_size=256):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.lstm1 = nn.LSTM(embed_size, hid_size, batch_first=True)

        self.lin  = nn.Linear(hid_size, 1)

        

    def forward(self, x):

        x = self.embed(x)

        x, (h_t, h_c) = self.lstm1(x)

        out = self.lin(h_t)

        return torch.sigmoid(out)
print(max(vocab.values()))
rnn = RNN(max(vocab.values())+1, 32, 10).cuda()

# +1 - padding(0)

optim = optimizer.Adam(rnn.parameters(), lr=2e-3)

crit  = nn.BCELoss()
epoches = 4
def train(model, trainloader, validloader, epoches, optim,

          crit):

    for epoche in range(epoches):

        rnn.train()

        for c,(xx, yy) in tqdm_notebook(enumerate(trainloader), total=len(trainloader)):

            xx, yy = xx.cuda(), yy.cuda()

            optim.zero_grad()

            out = model(xx)[0]

            loss = crit(out, yy)

            loss.backward()

            optim.step()

            if c % 150 == 0:

                print("Epoche {}    loss= {}".format(epoche, loss.item()))

        rnn.eval()

        y_pred = []

        y_true = []

        with torch.no_grad():

            for _,(xx, yy) in tqdm_notebook(enumerate(validloader), total=len(validloader)):

                    y_true.extend(yy.tolist())

                    xx, yy = xx.cuda(), yy.cuda()

                    out = model(xx)[0]

                    y_pred.extend([i[0]>0.5 for i in out.tolist()])

            print(classification_report(y_true, y_pred))
train(rnn, trainloader, validloader, epoches, optim, crit)
def predict(model, predloader):

    model.eval()

    y_pred = []

    y_true = []

    with torch.no_grad():

        for _,(xx, yy) in tqdm_notebook(enumerate(predloader), total=len(predloader)):

                y_true.extend(yy.tolist())

                xx, yy = xx.cuda(), yy.cuda()

                out = model(xx)[0]

                y_pred.extend([i[0]>0.5 for i in out.tolist()])

        print(classification_report(y_true, y_pred))

        

    
predict(rnn, testloader)
class CNN(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, 50)

        self.c1 = nn.Conv1d(50, 120, 1)#50

        self.p1 = nn.MaxPool1d(5)#10

        self.c2 = nn.Conv1d(120, 300, 1)#10

        self.p2 = nn.MaxPool1d(2)#50

        self.l1 = nn.Linear(3000, 1)

        

    def forward(self, x):

        x = self.embed(x)

        x = x.transpose(1,2)

        x = self.c1(x)

        x = torch.relu(self.p1(x))

        x = self.c2(x)

        x = torch.relu(self.p2(x))

        x = x.view(-1,3000)

        x = torch.sigmoid(self.l1(x))

        return x
cnn = CNN(max(vocab.values())+1).cuda()
optim_cnn = optimizer.Adam(cnn.parameters(), lr=2e-3)
def train_cnn(model, trainloader, validloader, epoches, optim,

          crit):

    for epoche in range(epoches):

        rnn.train()

        for c,(xx, yy) in tqdm_notebook(enumerate(trainloader), total=len(trainloader)):

            xx, yy = xx.cuda(), yy.cuda()

            optim.zero_grad()

            out = model(xx)

            loss = crit(out, yy)

            loss.backward()

            optim.step()

            if c % 150 == 0:

                print("Epoche {}    loss= {}".format(epoche, loss.item()))

        rnn.eval()

        y_pred = []

        y_true = []

        with torch.no_grad():

            for _,(xx, yy) in tqdm_notebook(enumerate(validloader), total=len(validloader)):

                    y_true.extend(yy.tolist())

                    xx, yy = xx.cuda(), yy.cuda()

                    out = model(xx)

                    y_pred.extend([i[0]>0.5 for i in out.tolist()])

            print(classification_report(y_true, y_pred))
train_cnn(cnn, trainloader, validloader, epoches, optim_cnn, crit)
def predict_cnn(model, predloader):

    model.eval()

    y_pred = []

    y_true = []

    with torch.no_grad():

        for _,(xx, yy) in tqdm_notebook(enumerate(predloader), total=len(predloader)):

                y_true.extend(yy.tolist())

                xx, yy = xx.cuda(), yy.cuda()

                out = model(xx)

                y_pred.extend([i[0]>0.5 for i in out.tolist()])

        print(classification_report(y_true, y_pred))

        

    
predict_cnn(cnn, testloader)
def predict_ans(cnn, rnn, predloader):

    cnn.eval()

    rnn.eval()

    y_pred = []

    y_true = []

    with torch.no_grad():

        for _,(xx, yy) in tqdm_notebook(enumerate(predloader), total=len(predloader)):

                y_true.extend(yy.tolist())

                xx, yy = xx.cuda(), yy.cuda()

                out1 = cnn(xx)

                out2 = rnn(xx)[0]

                out = (out1+out2)/2

                y_pred.extend([i[0]>0.5 for i in out.tolist()])

        print(classification_report(y_true, y_pred))

        

    
predict_ans(cnn, rnn, testloader)