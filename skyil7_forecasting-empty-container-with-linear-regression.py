import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/monthly-container-holding-of-ports-in-south-korea/container.csv', index_col=0)

data.head()
data = data[data['Harbor']!='East Sea, Mukho']

data_notKR = data[data['isKorean']==False]

data = data[data['isKorean']==True]

data = pd.merge(data, data_notKR, how='left', on=['Harbor', 'Date'])

data.drop(['isKorean_x', 'isKorean_y'], axis=1, inplace=True)

print(data.shape)

data.head()
! pip install adamp

from adamp import AdamP

import torch

import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing
device = 'cuda'



SEED = 777

torch.manual_seed(SEED)

if device == 'cuda':

  torch.cuda.manual_seed_all(SEED)
preds = []

labels = []

for harbor in data['Harbor'].unique():

    h_data = data[data['Harbor']==harbor]

    h_data.drop('Harbor', axis=1, inplace=True)

    x = h_data[h_data['Date']<'2019-12-31']

    x = x.drop('Date', axis=1)

    

    y = h_data[h_data['Date']<'2020-01-31']

    y = y.drop('Date', axis=1)

    y = y['Empty_40_x'] + y['Empty_40_y']

    y = y.iloc[1:]

    

    x = np.array(x)

    y = np.array(y)

    

    x_train = x[:x.shape[0]]

    y_train = y[:x.shape[0]]

    x_test = x[x.shape[0]-1]

    y_test = y[x.shape[0]-1]

    

    Scaler=preprocessing.StandardScaler()

    x_train = Scaler.fit_transform(x_train)

    x_test = Scaler.transform(x_test.reshape(1, x_train.shape[1]))

    

    x_train=torch.FloatTensor(x_train).to(device)

    y_train=torch.FloatTensor(y_train).to(device)

    x_test=torch.FloatTensor(x_test).to(device)

    

    lin = torch.nn.Linear(x_train.shape[1],1)

    torch.nn.init.xavier_uniform_(lin.weight)

    model = torch.nn.Sequential(lin).to(device)

    loss = torch.nn.MSELoss().to(device)

    optimizer = AdamP(model.parameters(), lr=1000)

    

    epochs = 300

    err_history = []

    for epoch in range(1, epochs+1):

        optimizer.zero_grad()

        hypothesis = model(x_train)

        cost = loss(hypothesis, y_train)

        cost.backward()

        err_history.append(cost.item())

        optimizer.step()

        

    with torch.no_grad():

        pred = model(x_test).detach().cpu().numpy()

    preds.append(pred)

    labels.append(y_test)
npr = []

err = []

for i in range(len(preds)):

    npr.append(float(preds[i][0]))

    err.append(float(np.abs(labels[i]-preds[i][0])))
print(data['Harbor'].unique())

print(labels)

print(npr)

print(err)

print()

err_rate = np.array(err)/labels * 100

print(err_rate)

print(np.mean(err_rate))
err_rate = np.array(err)/labels * 100

print(err_rate)

print(np.mean(err_rate))