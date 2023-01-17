import torch

import torch.nn as nn

import random

import pandas as pd

import numpy as np



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':

  torch.cuda.manual_seed_all(111)



train = pd.read_csv('../input/2020-ai-exam-fashionmnist-1/mnist_train_label.csv', header=None)

train
train[0].min()
from torch.utils.data import DataLoader, TensorDataset



xtrain = train.loc[:, 1:785]

ytrain = train.loc[:, 0]



print(xtrain.shape, ytrain.shape)



xtrain=np.array(xtrain)

ytrain=np.array(ytrain)



xtrain=torch.FloatTensor(xtrain).to(device)

ytrain=torch.LongTensor(ytrain).to(device)



dataset = TensorDataset(xtrain, ytrain)

dl = DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

random.seed(111)

torch.manual_seed(111)



epochs=15



model = nn.Linear(784, 10, bias=True).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)



for epoch in range(epochs+1):

  H = model(xtrain)

  cost = nn.functional.cross_entropy(H, ytrain)

  

  optimizer.zero_grad()

  cost.backward()

  optimizer.step()



  if epoch % 1 == 0:

    print('Epoch:{} Cost:{:.4f}'.format(epoch, cost.item()))



print('Finished')
test = pd.read_csv('../input/2020-ai-exam-fashionmnist-1/mnist_test.csv', header=None)

test
with torch.no_grad():

  xtest=test.loc[:,:]

  xtest=np.array(xtest)

  xtest=torch.FloatTensor(xtest).to(device)



  H = model(xtest)

  correct_prediction = torch.argmax(H, 1)



correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)



submit=pd.read_csv('../input/2020-ai-exam-fashionmnist-1/submission.csv')



for i in range(len(correct_prediction)):

  submit['Category'][i]=correct_prediction[i].item()



submit.to_csv('sub.csv',index=False, header=True)

submit