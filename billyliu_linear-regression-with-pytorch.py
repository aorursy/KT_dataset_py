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
import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader

# Import nn.functional

import torch.nn.functional as F
# Input (temp, rainfall, humidity)

inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 

                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 

                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 

                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 

                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 

                  dtype='float32')



# Targets (apples, oranges)

targets = np.array([[56, 70], [81, 101], [119, 133], 

                    [22, 37], [103, 119], [56, 70], 

                    [81, 101], [119, 133], [22, 37], 

                    [103, 119], [56, 70], [81, 101], 

                    [119, 133], [22, 37], [103, 119]], 

                   dtype='float32')



inputs = torch.from_numpy(inputs)

targets = torch.from_numpy(targets)

# Define dataset

train_ds = TensorDataset(inputs, targets)



# Define data loader

batch_size = 5

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
# Define model

model = nn.Linear(3, 2)

# print(model.weight)

# print(model.bias)
# Parameters

# list(model.parameters())



# Define loss function and loss

loss_fn = F.mse_loss

loss = loss_fn(model(inputs), targets)



# Define optimizer

opt = torch.optim.SGD(model.parameters(), lr=1e-5)
# Utility function to train the model

def fit(num_epochs, model, loss_fn, opt, train_dl):

    

    # Repeat for given number of epochs

    for epoch in range(num_epochs):

        

        # Train with batches of data

        for xb,yb in train_dl:

            

            # 1. Generate predictions

            pred = model(xb)

            

            # 2. Calculate loss

            loss = loss_fn(pred, yb)

            

            # 3. Compute gradients

            loss.backward()

            

            # 4. Update parameters using gradients

            opt.step()

            

            # 5. Reset the gradients to zero

            opt.zero_grad()

        

        # Print the progress

        if (epoch+1) % 10 == 0:

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



def main():

    fit(100, model, loss_fn, opt, train_dl)

if __name__ == '__main__':

    main()

    

    # Generate predictions

    preds = model(inputs)

    print(preds)



    # Compare with targets

    print(targets)