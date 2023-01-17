import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt # Importing matplotlib for plotting data

%matplotlib inline
import seaborn as sns
df = pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")
df.head()
df.describe()
inputs = df.drop("target_class", axis=1)

labels = df['target_class']
labels.head()
labels.hist(bins=2) # We can see that there is not a lot of pulsar stars so the neural network is more likely to predict that something is not a pulsar
inputs.head()
import torch

import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
len(inputs.columns)
class PulsarStarNN(nn.Module):

    def __init__(self):

        super(PulsarStarNN, self).__init__()

        self.fc1 = nn.Linear(8, 100)

        self.fc2 = nn.Linear(100, 100)

        self.fc3 = nn.Linear(100, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        return self.softmax(self.fc3(x))
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(inputs, labels, test_size=0.25)
train_X = torch.tensor(train_X.values).float()
train_X
test_X = torch.tensor(test_X.values).float()

train_y = torch.tensor(train_y.values).long()

test_y = torch.tensor(test_y.values).long()
train_y
net = PulsarStarNN()
net
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)
losses = []
for epoch in range(1, 201):

    optimizer.zero_grad()

    outputs = net(train_X)

    loss = criterion(outputs, train_y)

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    print("Epoch {}, Loss: {}".format(epoch, loss.item()))
plt.plot(losses)
pred_test = net(test_X)

_, preds_y = torch.max(pred_test, 1)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(test_y, preds_y)
print(classification_report(test_y, preds_y))
confusion_matrix(test_y, preds_y)