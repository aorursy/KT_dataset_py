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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
train_df.describe()
test_df.describe()
train_df['Sex'] = train_df['Sex'].apply(lambda x: 0 if x == "male" else 1)

test_df['Sex'] = test_df['Sex'].apply(lambda x: 0 if x == "male" else 1)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# For Higher quality images
train_df.hist(column="Age")
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
from sklearn.preprocessing import MinMaxScaler
import torch

import torch.nn as nn

import torch.nn.functional as f

import torch.optim as optim
from sklearn.model_selection import train_test_split
X = train_df[['Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age']]

y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.01, random_state=42, shuffle=True)
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

test_df = sc.transform(test_df[['Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age']])
X_train = torch.tensor(X_train).float()

X_test = torch.tensor(X_test).float()

y_train = torch.tensor(y_train.values).long()

y_test = torch.tensor(y_test.values).long()
X_train.shape
class TitanicSurvivorNN(nn.Module):

    def __init__(self):

        super(TitanicSurvivorNN, self).__init__()

        self.fc1 = nn.Linear(6, 100)

        self.fc2 = nn.Linear(100, 200)

        self.fc3 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 50)

        self.fc5 = nn.Linear(50, 2)

    def forward(self, x):

        x = f.relu(self.fc1(x))

        x = f.relu(self.fc2(x))

        x = f.relu(self.fc3(x))

        x = f.relu(self.fc4(x))

        return self.fc5(x)
model = TitanicSurvivorNN()

model
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())
losses = []

n_epochs = 450

for epoch in range(n_epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    print("Epoch {}, Loss {}".format(epoch, loss.item()))

    
fig = plt.figure(figsize=(14, 6))

plt.plot(losses)
test_X = torch.tensor(test_df).float()
test_out = model(test_X)
_, preds = torch.max(test_out, 1)
arr = []

for i in range(len(preds)):

    arr.append([892 + i, preds[i].item()])
sub_df = pd.DataFrame(arr)

sub_df.columns = ["PassengerId", "Survived"]

sub_df.head()
sub_df.to_csv("my_submission.csv", index=False)

print("Your submission was successfully saved!")
sub_df.hist(column="Survived", bins=2, )