%matplotlib inline

import os

import random



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torchvision



torch.set_default_tensor_type(torch.FloatTensor)
DATA_DIR = '/kaggle/input/house-prices-advanced-regression-techniques'

OUT_DIR = '/kaggle/working'

train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

idx = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[idx] = all_features[idx].apply(

    lambda x: (x - x.mean()) / (x.std())).fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)

test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)

train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
def get_k_fold_data(k, i, X, y):

    assert k > 1 and k > i

    fold_size = X.shape[0] // k

    # if i == 0:

    #     return X[fold_size:, :], y[fold_size:], X[:fold_size, :], y[:fold_size]

    return (

        torch.cat((X[:i*fold_size, :], X[(i + 1) * fold_size:, :]), dim=0),

        torch.cat((y[:i*fold_size], y[(i + 1) * fold_size:]), dim=0),

        X[i*fold_size:(i+1) * fold_size, :],

        y[i*fold_size:(i+1) * fold_size],

    )



def log_rmse(net: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:

    with torch.no_grad():

        # 将小于1的值设成1，使得取对数时数值更稳定

        clipped_preds = torch.max(net(features), torch.tensor(1.0))

        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())

    return rmse.item()
lr  = 5

k = 5

epochs = 100

batch_size = 64

weight_decay = 0.01

num_inputs = train_features.shape[1]

loss = nn.MSELoss()

def get_net(num_inputs: int):

    net = nn.Linear(num_inputs, 1)

    for param in net.parameters():

        nn.init.normal_(param, mean=0, std=0.1)

    return net
train_l_sum, valid_l_sum = 0.0, 0.0

net = get_net(num_inputs)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

for i in range(k):

    X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_features, train_labels)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_rmse, valid_rmse = list(), list()

    net = net.float()

    for epoch in range(epochs):

        for X, y in data_iter:

            l = loss(net(X.float()), y.float())

            optimizer.zero_grad()

            l.backward()

            optimizer.step()

        train_rmse.append(log_rmse(net, X_train, y_train))

        valid_rmse.append(log_rmse(net, X_valid, y_valid))

    train_l_sum += train_rmse[-1]

    valid_l_sum += valid_rmse[-1]

    print(f'fold {i}, train {train_rmse[-1]}, valid {valid_rmse[-1]}')

print(f'train {train_l_sum / k}, valid {valid_l_sum / k}')

print(f'std {torch.std(net(train_features) / train_labels - 1).detach().item()}')
predicts = net(test_features).detach().numpy()

test_data['SalePrice'] = pd.Series(predicts.reshape(1, -1)[0])

submission = pd.concat((test_data['Id'], test_data['SalePrice']), axis=1)

submission.to_csv(os.path.join(OUT_DIR, 'submission.csv'), index=False)