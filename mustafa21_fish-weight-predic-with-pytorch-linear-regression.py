

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import torch

from sklearn import preprocessing
df = pd.read_csv('../input/fish-market/Fish.csv')
df.shape
df['Species'].value_counts()
le = preprocessing.LabelEncoder()

le.fit(['Perch', 'Bream', 'Roach', 'Pike', 'Smelt', 'Parkki','Whitefish' ])
sp = df['Species']

sp[0]
list(le.classes_)

Species = le.transform(sp)
sp = Species.tolist()

df = df.drop('Species', axis=1)

df.shape
df.insert(0, 'sp', Species)

from sklearn.utils import shuffle

df = shuffle(df)
df.head()

labels = df['Weight']

dataset = df.drop('Weight',axis=1)
dataset.shape, labels.shape

labels = torch.tensor(labels.values.astype(np.float32))

dataset = torch.tensor(dataset.values.astype(np.float32))

labels = labels.unsqueeze(1)
dataset.shape, labels.shape
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset
train_d = TensorDataset(dataset, labels)

train_d[0:4]
batch_size = 50

train_dl = DataLoader(train_d, batch_size, shuffle=True)
import torch.nn as nn
model = nn.Linear(6, 1)

list(model.parameters())
preds = model(dataset)
import torch.nn.functional as F
loss_fn = F.mse_loss
loss = loss_fn(model(dataset), labels)

loss
opt = torch.optim.SGD(model.parameters(), lr=1e-5) 
def fit(num_epoches, model, loss_fn, opt):

    for epoch in range(num_epoches):

        for xb, yb in train_d:

            pred = model(xb)

            loss = loss_fn(pred, yb)

            loss.backward()

            opt.step()

            opt.zero_grad()

            

        if (epoch+1) % 10== 0:

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epoches, loss.item()))

        if loss.item() < 5 and loss.item() > 0:

            break
fit(200, model, loss_fn, opt)
preds = model(dataset)

preds[:10]
print(labels[:10])