# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head(5)
df_test.head(5)
def miss_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (total / data.isnull().count() * 100).sort_values(ascending = False)
    ms = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    ms = ms[ms["Total"] > 0]
    f, ax = plt.subplots(figsize = (8, 6))
    plt.xticks(rotation='90')
    fig = sns.barplot(ms.index, ms['Percent'], color='green', alpha = 0.8)
    plt.xlabel('Features', fontsize = 15)
    plt.ylabel('Percent of missing values', fontsize = 15)
    plt.title('Percent missing data by feature', fontsize = 15)
    return ms
miss_data(df_train)
miss_data(df_test)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace = True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace = True)
df_train.drop(['Cabin'], axis = 1, inplace = True)
df_test.drop(['Cabin'], axis = 1, inplace = True)
df_test['Age'].fillna(df_test['Age'].median(), inplace = True)
df_train['Age'].fillna(df_train['Age'].median(), inplace = True)
print (df_train.isnull().sum())
print (df_test.isnull().sum())
all_data = [df_train, df_test]
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
df_train.head()
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
for dataset in all_data:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins = [0, 12, 20, 40, 120], labels = ['Children', 'Teenage', 'Adult', 'Elder'])
for dataset in all_data:
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins = [0, 7.91, 14.45, 31, 120], labels = ['Low_fare', 'median_fare', 'fare', 'high_fare'])
train = df_train.copy()
test = df_test.copy()
all_dat = [train, test]
for dataset in all_dat:
    drop_column = ['Age', 'Fare', 'Name', 'Ticket']
    dataset.drop(drop_column, axis = 1, inplace = True)
train.head()
test.head()
cats = ['Sex', 'Title', 'Age_bin', 'Embarked', 'Fare_bin']
for cat in cats:
    train[cat] = train[cat].astype('category')
    test[cat] = test[cat].astype('category')
for cat in cats:
    train[cat] = train[cat].cat.codes
    test[cat] = test[cat].cat.codes
train.head()
test.head()
train.drop(['PassengerId'], axis = 1, inplace = True)
test.drop(['PassengerId'], axis = 1., inplace = True)
X = train.drop(['Survived'], axis = 1).values
y = train['Survived'].values
y = y.reshape(y.shape[0], 1)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
X_train.shape
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
net = Net()
opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss()
def train_epoch(model, opt, criterion, batch_size=50):
    model.train()
    losses = []
    for beg_i in range(0, X_train.size(0), batch_size):
        x_batch = X_train[beg_i:beg_i + batch_size, :]
        y_batch = y_train[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.numpy())
    return losses
e_losses = []
num_epochs = 200
for e in range(num_epochs):
    e_losses += train_epoch(net, opt, criterion)
plt.plot(e_losses)
net.eval()
y_pred = 1 * (net(X_test) >= 0.5)
accuracy_score(y_test, y_pred)
t = test.values
t = torch.from_numpy(t).type(torch.FloatTensor)
y_pred = 1 * (net(t) >= 0.5)
y_pred = np.array(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0], )
Res = pd.DataFrame()
Res['PassengerId'] = df_test['PassengerId']
Res['Survived'] = y_pred
Res.set_index(['PassengerId'], inplace=True)
Res.to_csv('prediction.csv', sep=',')
