import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
import sys

import pandas as pd

import random

from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(484)
torch.manual_seed(484)
if device == 'cuda':
  torch.cuda.manual_seed_all(484)
train = pd.read_csv('/kaggle/input/defense-project/train_data.csv')
train
scaler = StandardScaler()
test = pd.read_csv('/kaggle/input/defense-project/test_data.csv')

x_train_data = train.loc[:,'# mean_0_a':'fft_749_b']
y_train_data = train.loc[:,'label']

xs_data = scaler.fit_transform(x_train_data)
# train
x_train = torch.FloatTensor(xs_data[:1200])
x_vali = torch.FloatTensor(xs_data[1200:]) # Validation Data
y_train = torch.LongTensor(y_train_data[:1200].values)
y_vali = torch.LongTensor(y_train_data[1200:].values)
print(x_train.shape)
print(x_vali.shape)
l1 = torch.nn.Linear(2548,128) #딥러닝 모델로 수정
l2 = torch.nn.Linear(128, 128)
l3 = torch.nn.Linear(128, 128)
l4 = torch.nn.Linear(128, 128)
l4 = torch.nn.Linear(128, 3)
dr = torch.nn.Dropout(0.4)
relu = torch.nn.LeakyReLU()
torch.nn.init.xavier_uniform_(l1.weight)
torch.nn.init.xavier_uniform_(l2.weight)
torch.nn.init.xavier_uniform_(l3.weight)
torch.nn.init.xavier_uniform_(l4.weight)
model = torch.nn.Sequential(
    l1, relu, dr, 
    l2, relu, dr, 
    l3, relu, dr, 
    l4
).to(device)
model
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=100,
                                          shuffle=True,
                                          drop_last=False)
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
total_batch = len(data_loader)
model_history = []
err_history = []
vali_err_history = []
x_vali = x_vali.to(device)
y_vali = y_vali.to(device)
for epoch in range(1, 1+300):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        # 그래디언트 초기화
        optimizer.zero_grad()
        # Forward 계산
        hypothesis = model(X)
        # Error 계산
        cost = loss(hypothesis, Y)
        # Backparopagation
        cost.backward()
        # 가중치 갱신
        optimizer.step()

        # 평균 Error 계산
        avg_cost += cost
    avg_cost /= total_batch
    
    with torch.no_grad():
        pred = model(x_vali)
        torch.nn.CrossEntropyLoss().to(device)
        vali_err = loss(pred, y_vali)
        vali_err_history.append(vali_err)
    
    model_history.append(model)
    err_history.append(avg_cost)

    if epoch % 50 == 1 :
        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))

print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost))
import matplotlib.pyplot as plt

plt.plot(err_history[10:])
plt.plot(vali_err_history[10:])
plt.show()
best_model = model_history[np.argmin(vali_err_history)]
print(min(vali_err_history))
print(np.argmin(vali_err_history))
test = pd.read_csv('/kaggle/input/defense-project/test_data.csv')
test = test.to_numpy()
test = scaler.transform(test)
with torch.no_grad():
    x_test = torch.FloatTensor(test).to(device)
    
    pred = model(x_test)
    pred = pred.cpu()
    
    real_test_df = pd.DataFrame([[i, r] for i, r in enumerate(torch.argmax(pred, dim=1).numpy())], columns=['Id',  'Category'])
    real_test_df.to_csv('result.csv', mode='w', index=False)