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
import torch

torch.manual_seed(1234)

np.random.seed(1234)
pos_train = []

neg_train = []

pos_test = []

neg_test = []
for i in os.listdir("../input/aclimdb/aclImdb/train/pos/"):

    with open("../input/aclimdb/aclImdb/train/pos/" + i) as f:

        pos_train.append([f.read().split(), 1])

for i in os.listdir("../input/aclimdb/aclImdb/train/neg/"):

    with open("../input/aclimdb/aclImdb/train/neg/" + i) as f:

        neg_train.append([f.read().split(), 0])
for i in os.listdir("../input/aclimdb/aclImdb/train/pos/"):

    with open("../input/aclimdb/aclImdb/train/pos/" + i) as f:

        pos_test.append([f.read().split(), 1])

for i in os.listdir("../input/aclimdb/aclImdb/test/neg/"):

    with open("../input/aclimdb/aclImdb/test/neg/" + i) as f:

        neg_test.append([f.read().split(), 0])
import random

train = pos_train+neg_train

test = pos_test+neg_test

random.shuffle(train)
dictionary = {}

dictionary["#pad#"] = 0

dictionary["#unk#"] = 1

index = 2

with open("../input/aclimdb/aclImdb/imdb.vocab") as f:

    for i in f.read().split()[:10000]:

        dictionary[i.lower()] = index

        index += 1
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
for i in range(len(test)):

    for j in range(len(test[i][0])):

        s = test[i][0][j].lower()

        if s in dictionary:

            test[i][0][j] = dictionary[s]

        else:

            test[i][0][j] = dictionary["#unk#"]

    if len(test[i][0]) > 100:

        test[i][0] = test[i][0][:100]

    else:

        for k in range(100 - len(test[i][0])):

            test[i][0].append(dictionary["#pad#"])
for i in range(len(train)):

    for j in range(len(train[i][0])):

        s = train[i][0][j].lower()

        if s in dictionary:

            train[i][0][j] = dictionary[s]

        else:

            train[i][0][j] = dictionary["#unk#"]

    if len(train[i][0]) > 100:

        train[i][0] = train[i][0][:100]

    else:

        for k in range(100 - len(train[i][0])):

            train[i][0].append(dictionary["#pad#"])


class dataset(Dataset):

    def __init__(self, data):

        self.data = data

        

    def __getitem__(self, i):

        

        return torch.tensor(data=self.data[i][0]), torch.FloatTensor(data=[self.data[i][1]])

    

    def __len__(self):

        return len(self.data)
random.shuffle(test)

valid = train[:3000].copy()

del train[:3000]
ts = dataset(train)
tts = dataset(test)

vs = dataset(valid)
trainloader = DataLoader(ts, batch_size=30, shuffle=True)

validloader = DataLoader(vs, batch_size=30)

testloader = DataLoader(tts, batch_size=30)
class Net(nn.Module):

    def __init__(self, dict_size, pad):

        super().__init__()

        kernel_size = [2,3,4]

        self.e1 = nn.Embedding(dict_size, 30, padding_idx=pad)

        self.l1 = nn.LSTM(30, 150, batch_first=True)

        self.out = nn.Linear(150, 1)

        

    def forward(self, x):

        x = self.e1(x)

        x, (h_t, _) = self.l1(x)

        out = torch.sigmoid(self.out(h_t))

        return out
net = Net(len(dictionary), dictionary["#pad#"]).cuda()
import torch.optim as optim

opt = optim.Adam(net.parameters(), lr=0.003)

crit = nn.BCELoss()
from sklearn.metrics import classification_report

for i in range(5):

    num = 0

    net.train()

    for x, y in trainloader:

        opt.zero_grad()

        x, y = x.cuda(), y.cuda()

        out = net(x)[0]

        loss = crit(out, y)

        loss.backward()

        opt.step()

        if num % 50 == 0:

            print(num, " ", len(trainloader))

        num += 1

    net.eval()

    true = []

    outs = []

    with torch.no_grad():

        for x, y in validloader:

            x = x.cuda()

            out = net(x)[0]

            out1 = []

            for i in out:

                if i > 0.5:

                    out1.append(1)

                else: out1.append(0)

            outs.extend(out1)

            true.extend(y)

        print(classification_report(true, outs))
from sklearn.metrics import accuracy_score

net.eval()

true = []

outs = []

with torch.no_grad():

    for x, y in testloader:

        x = x.cuda()

        out = net(x)[0]

        out1 = []

        for i in out:

            if i >= 0.5:

                out1.append(1)

            else: out1.append(0)

        outs.extend(out1)

        true.extend(y)

    print(classification_report(true, outs))

    print(accuracy_score(true, outs))