import pandas as pd

import numpy as np

import torch

from torch import nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os

import cv2
df = pd.read_csv('../input/russian-handwritten-letters/all_letters_info.csv')

df.head()
base_path = '../input/russian-handwritten-letters/all_letters_image/all_letters_image/'
features = []

labels = []



for i, file in enumerate(df['file'].values):

    features.append(cv2.resize(cv2.imread(base_path + file), (28, 28)))

    labels.append(df['label'][i])



features = np.asarray(features)

labels = np.asarray(labels)
# normalize

features = features / 255.0
device = torch.device("cuda:0")
features = torch.from_numpy(features).to(device).type(torch.cuda.FloatTensor)

labels = torch.from_numpy(labels).to(device).type(torch.cuda.LongTensor)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
model = nn.Sequential(nn.Conv2d(28,14,1),

                      nn.ReLU(),

                      nn.Dropout2d(0.1),

                      nn.MaxPool2d(2),

                      nn.Flatten(),

                      nn.Linear(196, 124),

                      nn.Sigmoid(),

                      nn.Linear(124, 64),

                      nn.Sigmoid(),

                      nn.Linear(64, 34),

                      nn.LogSoftmax(dim=1))



model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.42, momentum=0.9)

loss_fn = nn.CrossEntropyLoss()
losses = []
for e in range(2100):

    out = model(X_train)

    loss = loss_fn(out, y_train)

    losses.append(loss)

    if(e % 350 == 0):

        preds_test = model(X_test)

        loss_test = loss_fn(preds_test, y_test)

        print('Epoch:{0}, Error-Loss:{1}'.format(e, loss.item()))

        print('Epoch:{0}, Error-Test-Loss:{1}'.format(e, loss.item()))

        print('------------------------------------------------------')



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
plt.plot(losses)