# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

from torchvision import transforms
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
np_train = train.values

np_test = test.values
tensor_train = torch.from_numpy(np_train)

tensor_test = torch.from_numpy(np_test)
train_loader = torch.utils.data.DataLoader(tensor_train, batch_size=128) 

test_loader = torch.utils.data.DataLoader(tensor_test, batch_size=128)
n_input = 784

n_dense_1 = 64

n_dense_2 = 64

n_dense_3 = 64

n_out = 10
model = nn.Sequential(

    

    # first hidden layer: 

    nn.Linear(n_input, n_dense_1), 

    nn.ReLU(), 

    

    # second hidden layer: 

    nn.Linear(n_dense_1, n_dense_2), 

    nn.ReLU(), 

    

    # third hidden layer: 

    nn.Linear(n_dense_2, n_dense_3), 

    nn.ReLU(), 

    nn.Dropout(),  

    

    # output layer: 

    nn.Linear(n_dense_3, n_out) 

)
cost_fxn = nn.CrossEntropyLoss() # includes softmax activation




optimizer = torch.optim.Adam(model.parameters())



def accuracy_pct(pred_y, true_y):

  _, prediction = torch.max(pred_y, 1) # returns maximum values, indices; fed tensor, dim to reduce

  correct = (prediction == true_y).sum().item()

  return (correct / true_y.shape[0]) * 100.0


n_batches = len(train_loader)

n_batches



n_epochs = 10 



print('Training for {} epochs. \n'.format(n_epochs))



for epoch in range(n_epochs):

  

  avg_cost = 0.0

  avg_accuracy = 0.0

  

  for i, (X, y) in enumerate(train_loader): # enumerate() provides count of iterations  

    

    # forward propagation:

    X_flat = X.view(X.shape[0], -1)

    y_hat = model(X_flat)

    cost = cost_fxn(y_hat, y)

    avg_cost += cost / n_batches

    

    # backprop and optimization via gradient descent: 

    optimizer.zero_grad() # set gradients to zero; .backward() accumulates them in buffers

    cost.backward()

    optimizer.step()

    

    # calculate accuracy metric:

    accuracy = accuracy_pct(y_hat, y)

    avg_accuracy += accuracy / n_batches

    

    if (i + 1) % 100 == 0:

      print('Step {}'.format(i + 1))

    

  print('Epoch {}/{} complete: Cost: {:.3f}, Accuracy: {:.1f}% \n'

        .format(epoch + 1, n_epochs, avg_cost, avg_accuracy)) 

  # TO DO: add test metrics



print('Training complete.')