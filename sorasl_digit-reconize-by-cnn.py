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

import pandas as pd

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
def opencsv():



    train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv", dtype=np.float32)



    # split data into features(pixels) and labels(numbers from 0 to 9)

    targets_numpy = train.label.values

    features_numpy = train.loc[:, train.columns != "label"].values / 255

    features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                                  targets_numpy,

                                                                                  test_size=0.2,

                                                                                  random_state=42)

    train_data = torch.from_numpy(features_train)

    train_label = torch.from_numpy(targets_train).type(torch.LongTensor)

    test_data = torch.from_numpy(features_test)

    test_label = torch.from_numpy(targets_test).type(torch.LongTensor)



    # print(train_data)

    return train_data, train_label, test_data, test_label
class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()



        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 16, 5, 1, 0),

            nn.ReLU(),

            nn.MaxPool2d(2),

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(16, 32, 5, 1, 0),

            nn.ReLU(),

            nn.MaxPool2d(2),

        )

        self.out = nn.Linear(32 * 4 * 4, 10)



    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        output = self.out(x)





        return output
trainData, trainLabel, testData, testLabel = opencsv()

batch_size = 100

n_iters = 2500

learning_rate = 0.004

num_epochs = n_iters / (len(trainData) / batch_size)

num_epochs = int(num_epochs)

# print(num_epochs)



train = TensorDataset(trainData, trainLabel)

test = TensorDataset(testData, testLabel)



# print(train, test)

trainLoader = DataLoader(train, batch_size=batch_size, shuffle=False)

testLoader = DataLoader(test, batch_size=batch_size, shuffle=False)



cnn = CNNModel()

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):

    for step, (x, y) in enumerate(trainLoader):



        train_x = x.view(100, 1, 28, 28)

        train_y = y

        # print(train_x)



        optimizer.zero_grad()

        output = cnn(train_x)

        loss = loss_func(output, train_y)

        loss.backward()

        optimizer.step()



        if step % 50 == 0:



            correct = 0

            total = 0

            for step_in, (test_x, test_y) in enumerate(testLoader):



                test_x = test_x.view(100, 1, 28, 28)

                testOutput = cnn(test_x)



                pred_y = torch.max(testOutput.data, 1)[1]

                total += len(test_y)

                correct += (pred_y == test_y).sum()



            accuracy = 100 * correct / float(total)

            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(step, loss.data, accuracy))

torch.save(cnn, './mnist_test.pt')



test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', dtype=np.float32).values.reshape(-1, 1, 28, 28) / 255

test_tensor = torch.from_numpy(test_data)

prediction_cnn = cnn(test_tensor)

prediction = (torch.max(prediction_cnn.data, 1)[1]).numpy()

np.savetxt("./submission.csv", np.dstack((np.arange(1, prediction.size+1), prediction))[0], "%d,%d", header="ImageId,Label", comments='')