# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data

import torch

import pandas as pd

from sklearn.model_selection import train_test_split



torch.random.manual_seed(123)



# Read the data

X_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', )

X_test_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', )

# X_full.head(10)



all_features = pd.concat((X_full.iloc[:, 1:-1], X_test_full.iloc[:, 1:]))



numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())

all_features = all_features.fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)

print(all_features.shape)



n_train = X_full.shape[0]



train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)

test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)

train_labels = torch.tensor(X_full.SalePrice.values, dtype=torch.float).view(-1, 1)

# define model

import torch

import torch.nn as nn





class HorsePrice(nn.Module):

    def __init__(self, in_features, mid_features=256):

        super().__init__()

        self.layer = nn.Sequential(

            nn.Linear(in_features, mid_features),

            nn.ReLU(),

            nn.Linear(mid_features, mid_features),

            nn.ReLU(),

            nn.Linear(mid_features, 1)

        )

        self.apply(self.weight_init)



    @staticmethod

    def weight_init(m):

        if isinstance(m, nn.Linear):

            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

#             nn.init.kaiming_normal_(m.weight)

            nn.init.constant_(m.bias, 0.1)



    def forward(self, x):

        return self.layer(x)
# train model

def train(model, train_features, train_labels, valid_features, valid_labels,

          nums_epoch, bs=64, lr=0.1, weight_decay=1e-5):

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, )

    optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.MSELoss()

    train_mae, valid_mae = [], []

    best_valid_mae = 1e10

    best_model = "best_model.pkl"

    for epoch in range(nums_epoch):

        model.train()

        for data, label in train_loader:

            out = model(data)

            loss = loss_fn(out, label)

            optim.zero_grad()

            loss.backward()

            optim.step()

        mae = torch.mean(torch.abs(model(train_features) - train_labels)).item()

#         print('[EPOCH {:03d}] TRAIN MAE'.format(epoch), mae)

        train_mae.append(mae)

        if valid_labels is not None:

            model.eval()

            mae = torch.mean(torch.abs(model(valid_features) - valid_labels)).item()

            valid_mae.append(mae)

#             print('[EPOCH {:03d}] VALID MAE'.format(epoch), mae)

            if mae < best_valid_mae:

                best_valid_mae = mae

                torch.save(model, best_model)

        else:

            torch.save(model, best_model)



    return best_valid_mae, best_model

# k fold validation



def k_cross_validation(k, train_data, train_label):

    assert k > 1

    nums_data = train_data.shape[0] // k

    validation_mae = []

    model_list = []

    for i in range(k):

        model = HorsePrice(354, 128)

        valid_features = train_data[i*nums_data: (i+1)*nums_data]

        valid_labels = train_label[i*nums_data: (i+1)*nums_data]



        train_features = torch.cat((train_data[:i*nums_data], train_data[(i+1)*nums_data:]))

        train_labels = torch.cat((train_label[:i*nums_data], train_label[(i+1)*nums_data:]))

        best_valid_mae, best_model = train(model, train_features, train_labels, valid_features, valid_labels, 15, lr=0.05, bs=32)

        validation_mae.append(best_valid_mae)

        model_list.append(torch.load(best_model))

        print('finish {} fold'.format(i))

    print('VALID MAE: ', validation_mae)

    return model_list

# train all data and submit

model_list = k_cross_validation(5, train_features, train_labels)

pred_prices = []

for m in model_list:

    m.eval()

    pred_price = m(test_features)

    pred_prices.append(pred_price.detach().numpy().flatten())

pred_prices = np.array(pred_prices)

out_df = pd.DataFrame({'Id': X_test_full.Id.values, 'SalePrice': pred_prices.mean(axis=0)})

out_df.to_csv('submission.csv', index=False)