# import necessary libraries

import os

import torch

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

os.listdir('../input/')

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline
# load the dataset

data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
# check for basic information

data.info()
# show first few recoreds

data.head()
# check for any missing values

data.isnull().sum()
# diagnosis distribution

data['diagnosis'].value_counts()
# encode categorical data

data['diagnosis'].replace({'M': 1, 'B': 0}, inplace = True)
Y = data['diagnosis'].to_numpy()
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1)
# split the dataset into the training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
# feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# create the model

model = torch.nn.Linear(X_train.shape[1], 1)
# load sets in format compatible with pytorch

X_train = torch.from_numpy(X_train.astype(np.float32))

X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train).float().reshape(-1, 1)

y_test = torch.from_numpy(y_test).float().reshape(-1, 1)
def configure_loss_function():

    return torch.nn.BCEWithLogitsLoss()
# use Adam optimiser for gradient descent

def configure_optimizer(model):

    return torch.optim.Adam(model.parameters(), lr = 0.0007)
# define the loss function to compare the output with the target

criterion = configure_loss_function()

optimizer = configure_optimizer(model)
# run the model

epochs = 2000

# initialise the train_loss & test_losses which will be updated

train_losses = np.zeros(epochs)

test_losses = np.zeros(epochs)



for epoch in range(epochs): 

    y_pred = model(X_train)

    loss = criterion(y_pred, y_train)

    # clear old gradients from the last step

    optimizer.zero_grad()

    # compute the gradients necessary to adjust the weights

    loss.backward()

    # update the weights of the neural network

    optimizer.step()



    outputs_test = model(X_test)

    loss_test = criterion(outputs_test, y_test)



    train_losses[epoch] = loss.item()

    test_losses[epoch] = loss_test.item()



    if (epoch + 1) % 50 == 0:

      print (str('Epoch ') + str((epoch+1)) + str('/') + str(epochs) + str(',  training loss = ') + str((loss.item())) + str(', test loss = ') + str(loss_test.item()))
# visualise the test and train loss

plt.plot(train_losses, label = 'train loss')

plt.plot(test_losses, label = 'test loss')

plt.legend()

plt.title('Model Loss')
with torch.no_grad():

  output_train = model(X_train)

  output_train = (output_train.numpy() > 0)



  train_acc = np.mean(y_train.numpy() == output_train)



  output_test = model(X_test)

  output_test = (output_test.numpy() > 0)

  

  test_acc = np.mean(y_test.numpy() == output_test)
print ('Train accuracy is: ' + str(train_acc))
print ('Test accuracy is: ' + str(train_acc))