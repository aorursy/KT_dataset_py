import numpy as np # linear algebra

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



torch.manual_seed(11)

np.random.seed(11)



train_dataframes = pd.read_csv("../input/train.csv")

test_dataframes = pd.read_csv("../input/test.csv")
MINI_BATCH = 10



X_train = [x[1:] for x in train_dataframes.values]

Y_train = [x[0] for x in train_dataframes.values]

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, random_state=0)

X_test = test_dataframes.values



X_train = torch.tensor(X_train, dtype=torch.float32)

X_validation = torch.tensor(X_validation, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

Y_train = torch.tensor(Y_train)

Y_validation = torch.tensor(Y_validation)



X_train = X_train / 128 - 1

X_validation = X_validation / 128 - 1

X_test = X_test / 128 - 1



X_train = X_train.view([-1, MINI_BATCH, 1, 28, 28])

X_validation = X_validation.view([-1, 1, 28, 28])

X_test = X_test.view([-1, 1, 28, 28])

Y_train = Y_train.view([-1, MINI_BATCH])
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self._conv1 = nn.Conv2d(1, 6, 5)

        self._pool = nn.MaxPool2d(2)

        self._conv2 = nn.Conv2d(6, 16, 5)

        self._fc1 = nn.Linear(256, 128)

        self._fc2 = nn.Linear(128, 64)

        self._fc3 = nn.Linear(64, 10)

    def forward(self, x):

        # m x 1 x 28 x 28

        x = self._conv1(x)

        x = F.relu(x)

        # m x 6 x 24 x 24

        x = self._pool(x)

        # m x 6 x 12 x 12

        x = self._conv2(x)

        x = F.relu(x)

        # m x 16 x 8 x 8

        x = self._pool(x)

        # m x 16 x 4 x 4

        x = x.view([-1, 256])

        # m x 640

        x = F.relu(self._fc1(x))

        x = F.relu(self._fc2(x))

        x = F.relu(self._fc3(x))

        return x
net = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



PRINT_EVERY = 10000 // MINI_BATCH

loss_plot_data = ([], [], [], [])

running_batches = 0

running_loss = 0

for epoch in range(0, 30):

    for i in range(0, len(X_train)):

        inputs = X_train[i]

        labels = Y_train[i]

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        running_batches += 1

        if running_batches % PRINT_EVERY == 0:

            validation_loss = 0

            with torch.no_grad():

                inputs = X_validation

                outputs = net(inputs)

                labels = Y_validation

                validation_loss = criterion(outputs, labels)

                

            print('[%d, %5d] loss: %.3f %.3f' % (epoch, i, running_loss / PRINT_EVERY, validation_loss))

            loss_plot_data[0].append((running_batches - 0.5 * PRINT_EVERY) * MINI_BATCH)

            loss_plot_data[1].append(running_loss / PRINT_EVERY)

            loss_plot_data[2].append(running_batches * MINI_BATCH)

            loss_plot_data[3].append(validation_loss)

            running_loss = 0

            

plt.plot(loss_plot_data[0], loss_plot_data[1])

plt.plot(loss_plot_data[2], loss_plot_data[3])

plt.show()
Y_test = net(X_test)

Y_test = [x.argmax().item() for x in Y_test ]



df = pd.DataFrame([(i + 1, Y_test[i]) for i in range(0, len(Y_test))], columns=['ImageId', 'Label'])



df.to_csv('submission.csv', index=None, header=True)