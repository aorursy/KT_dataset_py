import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



import matplotlib.pyplot as plt

%matplotlib inline
data_raw = pd.read_csv('../input/sign_mnist_train.csv', sep=",")

test_data_raw = pd.read_csv('../input/sign_mnist_test.csv', sep=",")
labels = data_raw['label']

data_raw.drop('label', axis=1, inplace=True)

labels_test = test_data_raw['label']

test_data_raw.drop('label', axis=1, inplace=True)
data = data_raw.values

labels = labels.values



test_data = test_data_raw.values

labels_test = labels_test.values


pixels = data[10].reshape(28, 28)

plt.subplot(221)

sns.heatmap(data=pixels)



pixels = data[12].reshape(28, 28)

plt.subplot(222)

sns.heatmap(data=pixels)



pixels = data[20].reshape(28, 28)

plt.subplot(223)

sns.heatmap(data=pixels)



pixels = data[32].reshape(28, 28)

plt.subplot(224)

sns.heatmap(data=pixels)
reshaped = []

for i in data:

    reshaped.append(i.reshape(1, 28, 28))

data = np.array(reshaped)



reshaped_test = []

for i in test_data:

    reshaped_test.append(i.reshape(1,28,28))

test_data = np.array(reshaped_test)
x = torch.FloatTensor(data)

y = torch.LongTensor(labels.tolist())



test_x = torch.FloatTensor(test_data)

test_y = torch.LongTensor(labels_test.tolist())
class Network(nn.Module): 

    

    def __init__(self):

        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 3)

        self.pool1 = nn.MaxPool2d(2)

        

        self.conv2 = nn.Conv2d(10, 20, 3)

        self.pool2 = nn.MaxPool2d(2)

        

        self.conv3 = nn.Conv2d(20, 30, 3) 

        self.dropout1 = nn.Dropout2d()

        

        self.fc3 = nn.Linear(30 * 3 * 3, 270) 

        self.fc4 = nn.Linear(270, 26)

        

        self.softmax = nn.LogSoftmax(dim=1)

    

    

    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = self.pool1(x)

        

        x = self.conv2(x)

        x = F.relu(x)

        x = self.pool2(x)

        

        x = self.conv3(x)

        x = F.relu(x)

        x = self.dropout1(x)

                

        x = x.view(-1, 30 * 3 * 3) 

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        

        return self.softmax(x)

    

    def test(self, predictions, labels):

        

        self.eval()

        correct = 0

        for p, l in zip(predictions, labels):

            if p == l:

                correct += 1

        

        acc = correct / len(predictions)

        print("Correct predictions: %5d / %5d (%5f)" % (correct, len(predictions), acc))

        

    def evaluate(self, predictions, labels):

                

        correct = 0

        for p, l in zip(predictions, labels):

            if p == l:

                correct += 1

        

        acc = correct / len(predictions)

        return(acc)
!pip install torchsummary

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network().to(device)

summary(model, (1, 28, 28))
net = Network()



optimizer = optim.SGD(net.parameters(),0.001, momentum=0.7)

loss_func = nn.CrossEntropyLoss()
loss_log = []

acc_log = []



for e in range(50):

    for i in range(0, x.shape[0], 100):

        x_mini = x[i:i + 100] 

        y_mini = y[i:i + 100] 

        

        optimizer.zero_grad()

        net_out = net(Variable(x_mini))

        

        loss = loss_func(net_out, Variable(y_mini))

        loss.backward()

        optimizer.step()

        

        if i % 1000 == 0:

            #pred = net(Variable(test_data_formated))

            loss_log.append(loss.item())

            acc_log.append(net.evaluate(torch.max(net(Variable(test_x[:500])).data, 1)[1], test_y[:500]))

        

    print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))
plt.figure(figsize=(10,8))

plt.plot(loss_log[2:])

plt.plot(acc_log)

plt.plot(np.ones(len(acc_log)), linestyle='dashed')

plt.show()
predictions = net(Variable(test_x))

net.test(torch.max(predictions.data, 1)[1], test_y)