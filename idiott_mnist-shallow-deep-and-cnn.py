!pip install -U torch
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import torch

print(torch.__version__)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.optim as optim



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

train_Y = mnist_train.iloc[:, 0].values

train_X = mnist_train.iloc[:, 1:].values

test_Y = mnist_test.iloc[:, 0].values

test_X = mnist_test.iloc[:, 1:].values
print(train_X.shape, train_Y.shape)

print(test_X.shape, test_Y.shape)
class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.dataframe = pd.read_csv(csv_file)

        self.transform = transform

        

    def __len__(self):

        return len(self.dataframe)

    

    def __getitem__(self, index):

        if torch.is_tensor(index):

            index = index.tolist()

            

        image = self.dataframe.iloc[index, 1:].values.astype('float32')

        #print(type(self.dataframe.iloc[index, 0]))

        label = self.dataframe.iloc[index, 0]

        image = image.reshape((28, 28))

        # Normalize TODO:use transform to do it

        image = (image - image.mean()) / image.std()

        return image, label

            

trainDataset = MNISTDataset("../input/mnist-in-csv/mnist_train.csv")

trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=10, shuffle=True)

testDataset = MNISTDataset("../input/mnist-in-csv/mnist_test.csv")

testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=10000)
plt.figure(figsize=(3, 3))

i = 87

plt.gca().invert_yaxis()

plt.pcolormesh(trainDataset[i][0])

print(trainDataset[i][1])
def fit(model, epochs):

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        overall_loss = 0

        for i, (X, Y) in enumerate(trainDataloader):

            Y_pred = model(X)

            loss = loss_fn(Y_pred, Y)

            overall_loss += loss.item()

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            if i % 1000 == 999:

                print(f'in epoch {epoch}, batch {i+1}, loss = {overall_loss / 1000}')

                overall_loss = 0.

            

def val(model):

    for X, Y in testDataloader:

        Y_pred = model(X).argmax(dim=1)

        rightCount = (Y == Y_pred).sum().item()

        accuracy = rightCount / 10000

        print("accuracy =", accuracy)

        
model = nn.Sequential(

    nn.Flatten(),

    nn.Linear(784, 10),

)

fit(model, 1)

val(model)
model = nn.Sequential(

    nn.Flatten(),

    nn.Linear(784, 64),

    nn.ReLU(),

    nn.Linear(64, 10),

)

fit(model, 1)

val(model)
class Preprocess(nn.Module):

    def __init__(self):

        super(Preprocess, self).__init__()

        

    def forward(self, x):

        return x.unsqueeze(1)
model = nn.Sequential(

    Preprocess(),

    nn.Conv2d(1, 32, 3),# 26 * 26 * 32

    nn.ReLU(),

    nn.MaxPool2d(2),# 13 * 13 * 32

    nn.Conv2d(32, 16, 5),# 9 * 9 * 16

    nn.ReLU(),

    nn.MaxPool2d(3),# 3 * 3 * 16

    nn.Conv2d(16, 10, 3),# 1 * 1 * 10

    nn.Flatten(),

)

fit(model, 1)

val(model)