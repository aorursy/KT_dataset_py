# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
p = df.hist(figsize=(20,20))
from sklearn.preprocessing import StandardScaler



df['amount_scaled'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df['time_scaled'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)
print('Number of fraud cases: {}, with percentage: {:.2f}%'.format(len(df[df.Class==1]), (len(df[df.Class==1])*100/len(df))))

print('Number of normal cases: {}, with percentage: {:.2f}%'.format(len(df[df.Class==0]), (len(df[df.Class==0])*100/len(df))))


missing_val_count_by_column = (df.isnull().sum())

num_cols_with_missing = len(missing_val_count_by_column[missing_val_count_by_column > 0])

print('Number of columns with missing values: ', num_cols_with_missing)
from sklearn.model_selection import train_test_split

import random



seed = 1

random.seed(seed)

np.random.seed(seed)

random_state=np.random.RandomState(seed)



X = df.drop('Class', axis=1)

y = df['Class']

X = X.values

y = y.values



idx_norm = y == 0

idx_out = y == 1

        

X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm], test_size=0.4, random_state=random_state)

X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out], test_size=0.4, random_state=random_state)

X_train = np.concatenate((X_train_norm, X_train_out))

y_train = np.concatenate((y_train_norm, y_train_out))



X_test = np.concatenate((X_test_norm, X_test_out))

y_test = np.concatenate((y_test_norm, y_test_out))

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



log_reg = LogisticRegression(penalty='l1', C=10, solver='liblinear')

log_reg.fit(X_train, y_train)



log_reg_pred = log_reg.predict(X_test)



fpr, tpr, thresold = roc_curve(y_test, log_reg_pred)

roc_auc = auc(fpr, tpr)



plt.figure(figsize=(9,7))

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve for imbalanced logistic regression')

plt.legend(loc="lower right")
from numpy.random import permutation



#now we do downsampling of the normal calss data

np.random.RandomState(seed)

perm = permutation(len(X_train_norm))

X_train_norm = X_train_norm[perm]

y_train_norm = y_train_norm[perm]

X_train_norm_downsmp = X_train_norm[:len(X_train_out)]

y_train_norm_downsmp = y_train_norm[:len(X_train_out)]





X_train_downsmp = np.concatenate((X_train_norm_downsmp, X_train_out))

y_train_downsmp = np.concatenate((y_train_norm_downsmp, y_train_out))



#log_reg = LogisticRegression(penalty='l1', C=10, solver='liblinear')

log_reg.fit(X_train_downsmp, y_train_downsmp)



log_reg_pred = log_reg.predict(X_test)



fpr, tpr, thresold = roc_curve(y_test, log_reg_pred)

roc_auc = auc(fpr, tpr)



plt.figure(figsize=(9,7))

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve for balanced logistic regression')

plt.legend(loc="lower right")
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import TensorDataset



input_dim = X_train.shape[1]



class AutoEncoder(nn.Module):

    

    def __init__(self):

        super(AutoEncoder, self).__init__()

        #encoder

        self.enc_layer1 = nn.Linear(input_dim,15)

        self.enc_layer2 = nn.Linear(15,10)

        #Decoder

        self.dec_layer1 = nn.Linear(10,15)

        self.dec_layer2 = nn.Linear(15,input_dim)

        

        

    def forward(self,x):

        x = F.relu(self.enc_layer1(x))

        x = F.relu(self.enc_layer2(x))

        x = F.relu(self.dec_layer1(x))

        x = F.relu(self.dec_layer2(x))

        

        return x



ae = AutoEncoder()

print(ae)
X_train_torch = torch.from_numpy(X_train).type(torch.FloatTensor) #note that we used X_train which contains data from both classes

y_train_torch = torch.from_numpy(y_train)



X_test_torch = torch.from_numpy(X_test).type(torch.FloatTensor)

y_test_torch = torch.from_numpy(y_test)



train = TensorDataset(X_train_torch,y_train_torch)

test = TensorDataset(X_test_torch,y_test_torch)



train_dataloader = torch.utils.data.DataLoader(train,batch_size=100,shuffle=True, num_workers=3)

test_dataloader = torch.utils.data.DataLoader(test,batch_size=50,shuffle=True, num_workers=3)
torch.manual_seed(seed)



loss_func = nn.MSELoss()

optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)



epochs = 10



#begin training

for epoch in range(epochs):

    for batch_idx, (data,target) in enumerate(train_dataloader):

        data = torch.autograd.Variable(data)

        optimizer.zero_grad()

        pred = ae(data)

        loss = loss_func(pred, data)

        loss.backward()

        optimizer.step()
ae.eval()

predictions = []

for batch_idx, (data,target) in enumerate(test_dataloader):

        data = torch.autograd.Variable(data)

        pred = ae(data)

        for prediction in pred:

            predictions.append(prediction.detach().numpy())

            

mse = np.mean(np.power(X_test - predictions, 2), axis=1)



fpr_ae, tpr_ae, thresold = roc_curve(y_test, mse)

roc_auc_ae = auc(fpr_ae, tpr_ae)



plt.figure(figsize=(9,7))

lw = 2

plt.plot(fpr_ae, tpr_ae, lw=lw, label='Autoencoder (AUC = %0.4f)' % roc_auc_ae)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve for autoencoder')

plt.legend(loc="lower right")

plt.show()
X_train_norm_torch = torch.from_numpy(X_train_norm).type(torch.FloatTensor)

y_train_norm_torch = torch.from_numpy(y_train_norm)



X_test_torch = torch.from_numpy(X_test).type(torch.FloatTensor)

y_test_torch = torch.from_numpy(y_test)



train = TensorDataset(X_train_norm_torch,y_train_norm_torch)

test = TensorDataset(X_test_torch,y_test_torch)



train_dataloader = torch.utils.data.DataLoader(train,batch_size=100,shuffle=True, num_workers=3)

test_dataloader = torch.utils.data.DataLoader(test,batch_size=50,shuffle=True, num_workers=3)



torch.manual_seed(seed)



ae = AutoEncoder()



loss_func = nn.MSELoss()

optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)



epochs = 10

for epoch in range(epochs):

    

    for batch_idx, (data,target) in enumerate(train_dataloader):

        data = torch.autograd.Variable(data)

        optimizer.zero_grad()

        pred = ae(data)

        loss = loss_func(pred, data)

        loss.backward()

        optimizer.step()

        

        

ae.eval()

predictions = []

for batch_idx, (data,target) in enumerate(test_dataloader):

        data = torch.autograd.Variable(data)

        pred = ae(data)

        for prediction in pred:

            predictions.append(prediction.detach().numpy())

            

mse = np.mean(np.power(X_test - predictions, 2), axis=1)

fpr_ae, tpr_ae, thresold = roc_curve(y_test, mse)

roc_auc_ae = auc(fpr_ae, tpr_ae)



plt.figure(figsize=(9,7))

lw = 2

plt.plot(fpr_ae, tpr_ae, lw=lw, label='Autoencoder (AUC = %0.4f)' % roc_auc_ae)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve for autoencoder')

plt.legend(loc="lower right")

plt.show()