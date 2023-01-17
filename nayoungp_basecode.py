import numpy as np

import torch

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
xy = pd.read_csv('../input/disneyland-crowd-levels/train.csv',header=None)

scaler = MinMaxScaler()

x_data = xy.loc[1:,1:6]

y_data = xy.loc[1:, 7:]

x_data=np.array(x_data, dtype=int)

y_data=np.array(y_data, dtype=int)

mintemp=x_data[:,0].min()

x_data[:,0]=x_data[:,0]-mintemp

scaler.fit(x_data)

x_data=scaler.transform(x_data)

x_train = torch.FloatTensor(x_data)

y_train = torch.LongTensor(y_data)

nb_class = 4

nb_data = len(y_train)

linear1 = torch.nn.Linear(6,100,bias=True)

linear2 = torch.nn.Linear(100,4,bias=True)

relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)

torch.nn.init.xavier_uniform_(linear2.weight)

dropout = torch.nn.Dropout(p=0.5)

model = torch.nn.Sequential(linear1,relu,dropout,

                            linear2)
y_train=y_train.reshape([761])

y_train=y_train-1
loss = torch.nn.CrossEntropyLoss()

op=optim.Adam(model.parameters(),lr=1e-3)

epochs=5000

minloss=1000

model=model.cuda()

loss=loss.cuda()

for epoch in range(epochs) : 

    rand= np.random.choice(np.array([i for i in range(761)]),761) 

    avg=[]

    op.zero_grad()

    output =model(x_train.cuda())

    cost=loss(output,y_train.cuda())

    cost.backward()

    op.step()

    avg.append(cost.item())

    if np.array(avg).mean()<minloss:

      minloss=np.array(avg).mean()

      goodmodel=model

    if(epoch%100==0):

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, minloss))
testxy = pd.read_csv('../input/disneyland-crowd-levels/test.csv', header=None)

tx_data = testxy.loc[1:,1:6]

tx_data=np.array(tx_data, dtype=int)

tx_data[:,0]=tx_data[:,0]-mintemp

tx_data=scaler.transform(tx_data)

x_test = torch.FloatTensor(tx_data)
with torch.no_grad():

  output=goodmodel(x_test.cuda())

  prediction = torch.argmax(output, dim=1)

  prediction=prediction+1
prediction
submission=pd.read_csv("../input/disneyland-crowd-levels/submission_sample.csv")

for i in range(len(prediction)):

    submission['Crowd level'][i] = prediction[i].item()

submission['Crowd level']=submission['Crowd level'].astype("int")

submission.to_csv("out.csv",index=False,header=True)
#정확도 : 58.620689655172406%