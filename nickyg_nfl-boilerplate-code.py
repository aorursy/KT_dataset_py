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
import numpy as np

import pandas as pd

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error

import gc

from tqdm import tqdm_notebook as tqdm

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
adjusted_group_df = pd.read_excel('./adjusted_group.xlsx')

adjusted_group_df.head()
split = int(round(len(adjusted_group_df)*.8,0))

test_adjusted_group_df = adjusted_group_df[split:]

adjusted_group_df = adjusted_group_df[:split]
class NFL_NN(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()



        self.fc1 = nn.Linear(in_features, 216)

        self.bn1 = nn.BatchNorm1d(216)

        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(216, 512)

        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 216)

        self.relu3 = nn.ReLU()

        self.dout3 = nn.Dropout(0.2)

        self.out = nn.Linear(216, out_features)

        self.out_act = nn.Sigmoid()

        

    def forward(self, input_):

        a1 = self.fc1(input_)

        bn1 = self.bn1(a1)

        h1 = self.relu1(bn1)

        a2 = self.fc2(h1)

        h2 = self.relu2(a2)

        a3 = self.fc3(h2)

        h3 = self.relu3(a3)

        dout3 = self.dout3(h3)

        a5 = self.out(dout3)

        y = self.out_act(a5)

        return a5
epoch = 10

batch_size = 1012
oof_crps_list = []

fold = GroupKFold(n_splits=5)





y = np.zeros(shape=(adjusted_group_df.shape[0], 199))

for i, yard in enumerate(adjusted_group_df['Yards'].values):

#     print(i, yard)

    y[i, yard+99:] = np.ones(shape=(1, 100-yard))



oof_preds = np.ones((len(adjusted_group_df), y.shape[1]))



feats = [

        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",

        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",

    ]



print('use feats: {}'.format(len(feats)))
for n_fold, (train_idx, valid_idx) in enumerate(fold.split(adjusted_group_df, y, groups=adjusted_group_df['GameId'])):

        print('Fold: {}'.format(n_fold+1))

        

        train_x, train_y = adjusted_group_df[feats].iloc[train_idx].values, y[train_idx]

        valid_x, valid_y = adjusted_group_df[feats].iloc[valid_idx].values, y[valid_idx] 



        train_x = torch.from_numpy(train_x)

        train_y = torch.from_numpy(train_y)

        valid_x = torch.from_numpy(valid_x)

        valid_y = torch.from_numpy(valid_y)



        train_dataset = TensorDataset(train_x, train_y)

        valid_dataset = TensorDataset(valid_x, valid_y)



        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        

        print('train: {}, valid: {}'.format(len(train_dataset), len(valid_dataset)))
in_features = adjusted_group_df[feats].shape[1]

out_features = y.shape[1]



model = NFL_NN(in_features, out_features)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)



for idx in range(10):

    print('Training epoch {}'.format(idx+1))

    train_batch_loss_sum = 0



    for param in model.parameters():

        param.requires_grad = True



    model.train()

    for x_batch, y_batch in tqdm(train_loader):

        

        y_pred = model(x_batch.float())

        loss = torch.sqrt(criterion(y_pred.float(), y_batch.view((len(y_batch), out_features)).float()))

        train_batch_loss_sum += loss.item()



#         del x_batch

#         del y_batch



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        torch.cuda.empty_cache()

        gc.collect()



    train_epoch_loss = train_batch_loss_sum / len(train_loader)



    model.eval()

    preds = np.zeros((len(valid_dataset), out_features))

    with torch.no_grad():

        for i, eval_x_batch in enumerate(valid_loader):

            eval_values = eval_x_batch[0].float()

            pred = model(eval_values)

            preds[i * batch_size:(i + 1) * batch_size] = pred



    valid_y_pred = preds

    valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))

    oof_preds[valid_idx] = valid_y_pred



    print('Train Epoch Loss: {:.5f}, Valid CRPS: {:.5f}'.format(train_epoch_loss, valid_crps))



# del model, criterion, optimizer

gc.collect()



print('DONE OOF ALL CRPS: {:.5f}'.format(np.sum(np.power(oof_preds - y, 2))/(199*len(oof_preds))))
def min_max_scaler(x):

    return (x - np.min(x)) / (np.max(x) - np.min(x))
feats = [

        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",

        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",

    ]

test = torch.from_numpy(test_adjusted_group_df[feats].values)

test_dataset = TensorDataset(test)

test_loader = DataLoader(test_dataset, batch_size, shuffle=False)



in_features = test_adjusted_group_df[feats].shape[1]

out_features = 199

# model = NFL_NN(in_features, out_features)

# model.load_state_dict(torch.load(model_path))



model.eval()

preds = np.zeros((len(test_dataset), out_features))

with torch.no_grad():

    for i, eval_x_batch in enumerate(test_loader):

        eval_values = eval_x_batch[0].float()

        pred = model(eval_values)

        preds[i * batch_size:(i + 1) * batch_size] = pred



y_pred = preds.copy()

adjust_preds = np.zeros((len(y_pred), y_pred.shape[1]))

for idx, pred in enumerate(y_pred):

    if idx==1: break

    prev = 0

    for i in range(len(pred)):

        if pred[i]<prev:

            pred[i]=prev

        prev=pred[i]

    x = min_max_scaler(pred)

    adjust_preds[idx, :] = x

    

adjust_preds[:, -1] = 1

adjust_preds[:, 0] = 0



preds_df = pd.DataFrame(data=adjust_preds.reshape(-1, 199))