import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# For plotting data

import plotly.plotly as py

import plotly.tools as tls

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
df.head()
df.info()

df.describe()
df[df.columns[2:]].std().plot('hist');

plt.title('Distribution of stds of all columns');
df[df.columns[2:]].mean().plot('hist');

plt.title('Distribution of means of all columns');
print('Distributions of first 28 columns')

plt.figure(figsize=(26, 24))

for i, col in enumerate(list(df.columns)[2:30]):

    plt.subplot(7, 4, i + 1)

    plt.hist(df[col])

    plt.title(col)
corrmat = df.corr()

abs(corrmat["target"][1:]).plot(kind='bar',stacked=True, figsize=(10,5))
bar = corrmat.loc['target'].max() * 0.08

to_drop = corrmat.loc['target'].index[corrmat.loc['target'] < bar]

train = df.drop(to_drop, 1)

train.head()
import torch

from torch import nn

from torch import optim

from torch.nn import functional as F
from sklearn.utils import shuffle

train = shuffle(train)

train.head()
train.shape
y = train.target

X = train.drop(['target'], axis=1)
feat = np.array(X)

target = np.array(y).reshape(250, 1)



feat = torch.from_numpy(feat).float().detach().requires_grad_(True)

target = torch.from_numpy(target).float().detach().requires_grad_(False)
feat_train = feat[:-40]

target_train = target[:-40]



feat_test = feat[-40:]

target_test = target[-40:]



feat_train.shape, target_test.shape
model = nn.Sequential(nn.Linear(31, 16),

                      nn.ReLU(),

                      nn.Linear(16, 1),

                      nn.Sigmoid())

model
opti = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.MSELoss() 
train_loss = []

test_loss  = []



train_acc = []

test_acc  = []
D = 20

for epoch in range(200):

    opti.zero_grad()

    pred = model(feat_train)

    

    loss = criterion(pred, target_train)

    

    loss.backward()

    opti.step()

    

    if not (epoch%D):

        train_loss.append(loss.item())

        

        pred = (pred > 0.5).float()

        acc  = pred == target_train

        train_acc.append(acc.sum().float()/len(acc))

        

    # Calculating the validation Loss

    with torch.no_grad():

        model.eval()

        pred = model(feat_test)

        tloss = criterion(pred, target_test)

        if not (epoch%D):

            test_loss.append(tloss.item())

            

            pred = (pred > 0.5).float()

            acc  = pred == target_test

            test_acc.append(acc.sum().float()/len(acc))

            print(F"{epoch:5d}  |  train accuracy: {train_acc[-1]:0.4f}  |  test accuracy: {test_acc[-1]:0.4f}  |  train loss: {train_loss[-1]:0.4f}  |  test loss: {test_loss[-1]:0.4f}")

    model.train()

            

print("DONE!")
plt.plot(train_loss, label='Training loss')

plt.plot(test_loss, label='Validation loss')

plt.legend(frameon=False)
plt.plot(train_acc, label='Training accuracy')

plt.plot(test_acc,  label='Validation accuracy')

plt.legend(frameon=False)
test = pd.read_csv('../input/test.csv')

test_id = test.id

test = test.drop(to_drop, axis=1)

final = np.array(test)

final = torch.from_numpy(final).float().requires_grad_(True)
ans = model(final) > 0.5
df = pd.DataFrame()

df['id'] = test_id

df['target'] = ans.detach().numpy().reshape(len(ans))

df[:10]
df.to_csv('Sollution.csv', index=False)