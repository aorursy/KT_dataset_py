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
import torch
import torch.nn as nn
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)
final_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype=np.float32)
sample_sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train.label.head()
train.info()

targets_np = train.label.values
features_np = train.loc[:, train.columns != 'label'].values 

examples_visualize = features_np[1:100].reshape(-1,28,28)
targets_visualize = targets_np[1:100]
rows, columns = 8, 8
fig = plt.figure(figsize=(rows*2,columns*2))
for i in range(rows*columns):
    plt.subplot(rows, columns,i+1)
    plt.tight_layout()
    plt.imshow(examples_visualize[i], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(targets_visualize[i]))
plt.show()

dict_ids = {}
for idx, value in enumerate(features_np ):
    id = targets_np[idx]    
    if id not in dict_ids:
        dict_ids[id] = 1
    else:
        dict_ids[id] += 1
clasess = list(dict_ids)
objects = (0,1,2,3,4,5,6,7,8,9)
y_pos = np.arange(len(objects))
performance = []
for i in clasess:
    performance.append(dict_ids[i])
fig = plt.figure(figsize=(10,10))

plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Classes id  usage')

plt.show()
targets_np = train.label.values
features_np = train.loc[:, train.columns != 'label'].values / 255.0
features_train, features_val, target_train, target_val = train_test_split(features_np, targets_np, test_size=0.2, random_state=42)

featuresTrain = torch.from_numpy(features_train.reshape(-1,1, 28,28))
targetsTrain = torch.from_numpy(target_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresVal = torch.from_numpy(features_val.reshape(-1,1,28,28))
targetsVal = torch.from_numpy(target_val).type(torch.LongTensor) # data type is long
featuresVal.shape
batch_size = 256

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
val = torch.utils.data.TensorDataset(featuresVal,targetsVal)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
train_loader
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
lr = 0.1
gamma = 0.7

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
epochs = 10
log_interval = 10
train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss, correct, len(val_loader.dataset),
                100. * correct / len(val_loader.dataset)))
    scheduler.step()
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend(frameon=False)
final_test
final_test_np = final_test.values/255

test_tn = torch.from_numpy(final_test_np)
rows, columns = 8, 8
fig = plt.figure(figsize=(rows*2,columns*2))
for i in range(rows*columns):
    output = model(test_tn[i].reshape(1,1,28,28))
    pred = output.argmax(dim=1, keepdim=True)
    plt.subplot(rows, columns,i+1)
    plt.tight_layout()
    plt.imshow(test_tn[i].numpy().reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predit image: {}".format(pred[0][0]))
plt.show()




