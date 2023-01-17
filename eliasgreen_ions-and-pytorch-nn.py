import numpy as np 

import pandas as pd

import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from torch.utils.data import DataLoader, Dataset

from torch.nn import functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score

import tqdm

import sys

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from IPython.display import clear_output

import itertools



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

torch.manual_seed(3246)

np.random.seed(3246)
submisson = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')



#df = df.head(1000)
df['time'] = (df['time'] - df['time'].mean()) / (df['time'].max() - df['time'].min())

df['signal'] = (df['signal'] - df['signal'].mean()) / (df['signal'].max() - df['signal'].min())
test_df['time'] = (test_df['time'] - test_df['time'].mean()) / (test_df['time'].max() - test_df['time'].min())

test_df['signal'] = (test_df['signal'] - test_df['signal'].mean()) / (test_df['signal'].max() - test_df['signal'].min())
submisson.head(2)
test_df.head(2)
df.head(2)
#one_hot = pd.get_dummies(df['open_channels'])

#df = df.drop('open_channels',axis = 1)

#df = df.join(one_hot)
df.head(2)
ROW_PER_BATCH = 500000

df_for_eda = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
df_for_eda['batch'] = 0



for i in range(0, df_for_eda.shape[0]//ROW_PER_BATCH):

    df_for_eda.iloc[i * ROW_PER_BATCH: (i+1) * ROW_PER_BATCH,3] = i
class oversampTrainData(Dataset):

    def __init__(self, data):

            self.data = torch.FloatTensor(data.values.astype('float'))



    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

            target = self.data[index][-1:]

            data_val = self.data[index][:-1]

            #list_target = list(0 for i in range(11)) 

            #list_target[int(target.item())] = 1

            #print(data_val)

            #print(target)

            return data_val, target
class oversampTestData(Dataset):

    def __init__(self, data):

            self.data = torch.FloatTensor(data.values.astype('float'))



    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

            data_val = self.data[index]

            return data_val
Train_Batch_Size = 128

Test_Batch_Size = 128



train_dataset = oversampTrainData(df)

test_dataset = oversampTestData(test_df)



device = "cuda" if torch.cuda.is_available() else "cpu"

kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

train_loader = DataLoader(train_dataset, batch_size=Train_Batch_Size, shuffle=True, **kwargs)

test_loader = DataLoader(test_dataset, batch_size=Test_Batch_Size, shuffle=False, **kwargs)
class LinearModel(nn.Module):

    def __init__(self):

        super(LinearModel, self).__init__()

        self.fc1 = nn.Linear(2, 1000)

        self.fc2 = nn.Linear(1000, 1000)

        self.fc3 = nn.Linear(1000, 1000)

        self.fc4 = nn.Linear(1000, 1000)

        self.fc5 = nn.Linear(1000, 1000)

        self.fc6 = nn.Linear(1000, 1000)

        self.fc7 = nn.Linear(1000, 1000)

        self.fc8 = nn.Linear(1000, 500)

        self.fc9 = nn.Linear(500, 500)

        self.fc10 = nn.Linear(500, 100)

        self.fc11 = nn.Linear(100, 11)

        

        

    def forward(self,x):

        out = self.fc1(x)

        out = F.relu(out)

        

        out = self.fc2(out)

        out = F.relu(out)

        

        out = self.fc3(out)

        out = F.relu(out)

        

        out = self.fc4(out)

        out = F.relu(out)

        

        out = self.fc5(out)

        out = F.relu(out)

        

        out = self.fc6(out)

        out = F.relu(out)

        

        out = self.fc7(out)

        out = F.relu(out)

        

        out = self.fc8(out)

        out = F.relu(out)

        

        out = self.fc9(out)

        out = F.relu(out)

        

        out = self.fc10(out)

        out = F.relu(out)

        

        out = self.fc11(out)

        #out = torch.softmax(out, dim=0)

        



        return out

    

model = LinearModel()



use_gpu = torch.cuda.is_available()

if use_gpu:

	model = model.cuda()

	print ('USE GPU')

else:

	print ('USE CPU')
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

scheduler = ReduceLROnPlateau(optimizer, 'max')
def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']
NUMBER_OF_EPOCHS = 5

f1_score_history = []

loss_history = []



for epoch in range(NUMBER_OF_EPOCHS): 

    epoch_loss = 0

    f1_score_val = -1

    

    

    model.train()

    

    for i in tqdm.tqdm(train_loader, position=0):

        data, target = i

        data, target = data.to(device), target.to(device)

        

        target_pred = model(data)

        

        target = target.squeeze()

        target = target.type(torch.LongTensor).to(device)

        

        loss = criterion(target_pred, target)

        epoch_loss += loss.item()



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    model.eval()

    

    pure_target_for_f1_score = []

    pred_target_for_f1_score = []

    

    for i in tqdm.tqdm(train_loader, position=0):

        data, target = i

        data, target = data.to(device), target.to(device)

        

        target_pred = model(data)

        

        target = target.squeeze()

        target = target.type(torch.LongTensor).to(device)

        

        #print('target', target.cpu().data.numpy().astype(int))

        #print('preds', np.argmax(target_pred.cpu().data.numpy(), axis=1))

        

        pure_target_for_f1_score += list(target.cpu().data.numpy().astype(int))

        pred_target_for_f1_score += list(np.argmax(target_pred.cpu().data.numpy(), axis=1))

    

    

    clear_output(wait=True)

    

    f1_score_val = f1_score(pure_target_for_f1_score, pred_target_for_f1_score, zero_division=1, average='macro')

    scheduler.step(f1_score_val)

    

    f1_score_history.append(f1_score_val)

    loss_history.append(epoch_loss/len(train_loader))

    

    plt.plot(loss_history)

    plt.plot(f1_score_history)

    

    plt.subplot(1, 2, 1)

    plt.plot(loss_history, 'ko-')

    plt.xlabel('epoch')

    plt.ylabel('LOSS')

    



    plt.subplot(1, 2, 2)

    plt.plot(f1_score_history, 'r.-')

    plt.xlabel('epoch')

    plt.ylabel('F1')

    

    plt.tight_layout()

    plt.pause(0.1)

        

    print('epoch: ', epoch,' loss: ', epoch_loss/len(train_loader), ' F1 score: ', f1_score_val, ' learning rate: ', get_lr(optimizer))
preds_for_subm = []

for data in tqdm.tqdm(test_loader, position=0):

        data = data.to(device)

        target_pred = model(data)

        

        preds_for_subm += list(np.argmax(target_pred.cpu().data.numpy(), axis=1))

        #print('preds', list(itertools.chain(*target_pred.cpu().data.numpy().astype(int))))

        

#print(preds_for_subm[:100])
submisson['open_channels'] = preds_for_subm

submisson['time'] = submisson['time'].astype(float)

submisson.to_csv('our_submission1.csv', index=False, float_format='%.4f')