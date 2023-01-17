import torch

import torch.optim as optim

import torch.nn.functional as F

import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler # For Normalization
device = torch.device("cuda")
torch.manual_seed(484)

torch.cuda.manual_seed_all(484)
# 데이터 불러오기. 환경에 따라 다를 수 있습니다.

dir = '/kaggle/input/lol-prediction/'



x_train = pd.read_csv(dir+'lol.x_train.csv', index_col=0)

y_train = pd.read_csv(dir+'lol.y_train.csv', index_col=0)

x_test = pd.read_csv(dir+'lol.x_test.csv', index_col=0)
x_train.head()
y_train.head()
x_train = np.array(x_train)

x_test = np.array(x_test)
scaler = MinMaxScaler() # Normalizer

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) # Not Fit Transform!



pd.DataFrame(x_train).describe()
y_train = np.array(y_train)
x_train = torch.FloatTensor(x_train)

y_train = torch.FloatTensor(y_train)

x_test = torch.FloatTensor(x_test)
from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader
# 커스텀 데이터 셋

train_set = TensorDataset(x_train, y_train)
# 10000 크기의 배치로 나누어 학습시킴

data_loader = DataLoader(dataset=train_set,

                         batch_size=10000,

                         shuffle=True)
# DNN 모델 구축

l1 = torch.nn.Linear(48, 32).to(device)

l2 = torch.nn.Linear(32, 1).to(device)

relu = torch.nn.ReLU()

sigmoid = torch.nn.Sigmoid()



model = torch.nn.Sequential(l1, relu, l2, sigmoid)

model
cost = torch.nn.BCELoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 60

for epoch in range(1, epochs+1):

    avg_cost = 0

    total_batch = len(data_loader)



    for x, y in data_loader:    # batch loop

        x = x.to(device)

        y = y.to(device)



        optimizer.zero_grad()

        hypothesis = model(x)

        cost_val = cost(hypothesis, y)

        cost_val.backward()

        optimizer.step()



        avg_cost += cost_val

    

    avg_cost /= total_batch



    if epoch % 10 == 1 or epoch == epochs:

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, epochs, avg_cost.item()))
with torch.no_grad(): # Don't Calculate Gradient

    x_test = x_test.to(device)



    pred = model(x_test)
pred[pred>=0.5] = 1.0

pred[pred<=0.5] = 0.0

pred = pred.detach().cpu().numpy()

pred = pred.astype(np.uint32)

id=np.array([i for i in range(pred.shape[0])]).reshape(-1, 1).astype(np.uint32)

result=np.hstack([id, pred])



submit = pd.DataFrame(result, columns=['id', 'blueWins'])

submit.to_csv('lol.baseline.csv', index=False)