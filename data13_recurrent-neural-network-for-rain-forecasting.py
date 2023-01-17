import os

import torch

import numpy as np

import pandas as pd

import seaborn as sns

from torch import nn, optim

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn import preprocessing

import torch.nn.functional as func

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
%matplotlib inline
sns.set(style='darkgrid')

sns.set_palette('deep')
# load the dataset

df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
# show first few records

df.head()
# show dataset dimensions

df.shape
# show dataset summary

df.info()
# show the frequency distribution of RainTomorrow

df['RainTomorrow'].value_counts()
# show percentage

df['RainTomorrow'].value_counts()/len(df)
df.isnull().sum()
numerical = ['Temp9am', 'MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'WindSpeed9am']

df[numerical].hist()
df[numerical].describe()
# fill missing values of normally-distributed columns with mean and skewed distribution with median

df['Temp9am'] = df['Temp9am'].fillna(value = df['Temp9am'].mean())

df['MinTemp'] = df['MinTemp'].fillna(value = df['MinTemp'].mean())

df['MaxTemp'] = df['MaxTemp'].fillna(value = df['MaxTemp'].mean())

df['Rainfall'] = df['Rainfall'].fillna(value = df['Rainfall'].mean())

df['Humidity9am'] = df['Humidity9am'].fillna(value = df['Humidity9am'].median())

df['WindSpeed9am'] = df['WindSpeed9am'].fillna(value = df['WindSpeed9am'].median())
df['RainToday'] = df['RainToday'].fillna(value = df['RainToday'].mode()[0])
# convert data variable into dattime type

df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# extract year from the date

df['Year'] = df['Date'].dt.year
# extract month from the date

df['Month'] = df['Date'].dt.month
# extract day from the date

df['Day'] = df['Date'].dt.day
# encode location

le = preprocessing.LabelEncoder()

df['Location'] = le.fit_transform(df['Location'])
# encode RainToday & RainTomorrow

df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)

df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)
X = df[['Temp9am', 'MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'WindSpeed9am', 'RainToday', 'Location', 'Year', 'Month', 'Day']]

y = df[['RainTomorrow']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train = torch.from_numpy(X_train.to_numpy()).float()

y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.from_numpy(X_test.to_numpy()).float()

y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# create the model

class Model(nn.Module):

  def __init__(self, n_features):

    super(Model, self).__init__()

    self.fc1 = nn.Linear(n_features, 11)

    self.fc2 = nn.Linear(11, 8)

    self.fc3 = nn.Linear(8, 5)

    self.fc4 = nn.Linear(5, 3)

    self.fc5 = nn.Linear(3, 1)

  def forward(self, x):

    x = func.relu(self.fc1(x))

    x = func.relu(self.fc2(x))

    x = func.relu(self.fc3(x))

    x = func.relu(self.fc4(x))

    return torch.sigmoid(self.fc5(x))
model = Model(X_train.shape[1])
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr = 0.001)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

X_train = X_train.to(device)

y_train = y_train.to(device)



X_test = X_test.to(device)

y_test = y_test.to(device)



model = model.to(device)
# define the loss function to compare the output with the target

criterion = criterion.to(device)
def calculate_accuracy(y_true, y_pred):

  predicted = y_pred.ge(.5).view(-1)

  return (y_true == predicted).sum().float() / len(y_true)
def round_tensor(t, decimal_places = 3):

  return round(t.item(), decimal_places)
# run the model

for epoch in range(1000):

    y_pred = model(X_train)

    y_pred = torch.squeeze(y_pred)

    train_loss = criterion(y_pred, y_train)

    if epoch % 100 == 0:

      train_acc = calculate_accuracy(y_train, y_pred)

      y_test_pred = model(X_test)

      y_test_pred = torch.squeeze(y_test_pred)

      test_loss = criterion(y_test_pred, y_test)

      test_acc = calculate_accuracy(y_test, y_test_pred)

      print (str('epoch ') + str(epoch) + str(' Train set: loss: ') + str(round_tensor(train_loss)) + str(', accuracy: ') + str(round_tensor(train_acc)) + str(' Test  set: loss: ') + str(round_tensor(test_loss)) + str(', accuracy: ') + str(round_tensor(test_acc)))

    optimiser.zero_grad()

    train_loss.backward()

    optimiser.step()
classes = ['No rain', 'Raining']



y_pred = model(X_test)

y_pred = y_pred.ge(.5).view(-1).cpu()

y_test = y_test.cpu()

print(classification_report(y_test, y_pred, target_names=classes))
conf_mat = confusion_matrix(y_test, y_pred)

df_conf_mat = pd.DataFrame(conf_mat, index = classes, columns = classes)

heat_map = sns.heatmap(df_conf_mat, annot = True, fmt = 'd')

heat_map.yaxis.set_ticklabels(heat_map.yaxis.get_ticklabels(), ha = 'right')

heat_map.xaxis.set_ticklabels(heat_map.xaxis.get_ticklabels(), ha = 'right')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')