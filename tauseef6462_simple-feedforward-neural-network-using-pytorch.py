import numpy as np

import pandas as pd        # For loading and processing the dataset

from sklearn.model_selection import train_test_split

import torch

import math

import matplotlib.pyplot as plt

import matplotlib.colors

import torch.nn.functional as F
# Read the CSV input file and show first 5 rows

df_train = pd.read_csv('../input/train.csv')

df_train.head(5)
# We can't do anything with the Name, Ticket number, and Cabin, so we drop them.

df_train = df_train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
# To make 'Sex' numeric, we replace 'female' by 0 and 'male' by 1

df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(int)
# We replace 'Embarked' by three dummy variables 'Embarked_S', 'Embarked_C', and 'Embarked Q',

# which are 1 if the person embarked there, and 0 otherwise.

df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)

df_train = df_train.drop('Embarked', axis=1)
# We normalize the age and the fare by subtracting their mean and dividing by the standard deviation

age_mean = df_train['Age'].mean()

age_std = df_train['Age'].std()

df_train['Age'] = (df_train['Age'] - age_mean) / age_std



fare_mean = df_train['Fare'].mean()

fare_std = df_train['Fare'].std()

df_train['Fare'] = (df_train['Fare'] - fare_mean) / fare_std
# In many cases, the 'Age' is missing - which can cause problems. Let's look how bad it is:

print("Number of missing 'Age' values: {:d}".format(df_train['Age'].isnull().sum()))



# A simple method to handle these missing values is to replace them by the mean age.

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
# With that, we're almost ready for training

df_train.head()
# Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and test set

X_train = df_train.drop('Survived', axis=1).as_matrix()

y_train = df_train['Survived'].as_matrix()



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
print(X_train.shape, y_train.shape)
def accuracy(y_hat, y):

  pred = torch.argmax(y_hat, dim=1)

  return (pred == y).float().mean()
X_train = X_train.float()

y_train = y_train.long()

X_test = X_test.float()

y_test = y_test.long()
import torch.nn as nn

from torch import optim
class FirstNetwork_v3(nn.Module):

  

  def __init__(self):

    super().__init__()

    torch.manual_seed(0)

    self.net = nn.Sequential(

        nn.Linear(9, 128), 

        nn.ReLU(),

        nn.Linear(128, 256),

        nn.ReLU(),

        nn.Linear(256, 2), 

        nn.Softmax()

    )



  def forward(self, X):

    return self.net(X)



  def predict(self, X):

    Y_pred = self.forward(X)

    return Y_pred
def fit_v2(x, y, model, opt, loss_fn, epochs = 1000):

  

  for epoch in range(epochs):

    loss = loss_fn(model(x), y)

    loss.backward()

    opt.step()

    opt.zero_grad()

    

  return loss.item()
device = torch.device("cuda")



X_train=X_train.to(device)

y_train=y_train.to(device)

X_test=X_test.to(device)

y_test=y_test.to(device)

fn = FirstNetwork_v3()

fn.to(device)

loss_fn = F.cross_entropy

opt = optim.SGD(fn.parameters(), lr=0.5)



print('Final loss', fit_v2(X_train, y_train, fn, opt, loss_fn))
Y_pred_train = fn.predict(X_train)

#Y_pred_train = np.argmax(Y_pred_train,1)

Y_pred_val = fn.predict(X_test)

#Y_pred_val = np.argmax(Y_pred_val,1)



accuracy_train = accuracy(Y_pred_train, y_train)

accuracy_val = accuracy(Y_pred_val, y_test)
print("Training accuracy", (accuracy_train))

print("Validation accuracy",(accuracy_val))
df_test = pd.read_csv('../input/test.csv')

df_test.head()
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Sex'] = df_test['Sex'].map({'female':0, 'male':1}).astype(int)

df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)

df_test = df_test.drop('Embarked', axis=1)

df_test['Age'] = (df_test['Age'] - age_mean) / age_std

df_test['Fare'] = (df_test['Fare'] - fare_mean) / fare_std

df_test.head()

X_test = df_test.drop('PassengerId', axis=1).as_matrix()
X_test = torch.tensor(X_test)
X_test = X_test.float()
X_test=X_test.to(device)
X_test.shape
Y_pred_val = fn.predict(X_test)
pred = torch.argmax(Y_pred_val, dim=1)
pred.shape
df_test['Survived'] = pred.to('cpu')
output = pd.DataFrame()

output['PassengerId'] = df_test['PassengerId']

output['Survived'] = df_test['Survived'].astype(int)

output.to_csv('./prediction.csv', index=False)

output.head()