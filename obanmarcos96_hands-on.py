# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch
import torchvision
# Preparo hiperparámetros
n_epochs = 10
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
p_dropout = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
train_target = torch.tensor(train['label'].values)
train_features = torch.tensor(train.filter(regex = 'pix').values.reshape(train.shape[0],1,28,28))

test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
test_target = torch.tensor(test['label'].values)
test_features = torch.tensor(test.filter(regex = 'pix').values.reshape(test.shape[0],1,28,28))
training = data.TensorDataset(train_features, train_target)
train_loader = data.DataLoader(training, batch_size = batch_size_train, shuffle = True)

testing = data.TensorDataset(test_features, test_target)
test_loader = data.DataLoader(testing, batch_size = batch_size_test, shuffle = True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# heredo las propiedades de la clase module
class Red(nn.Module):
    
    def __init__(self):
        
        super(Red, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        # regularización (preguntar al respecto)
        self.conv2_drop = nn.Dropout2d()
        self.linear1 = nn.Linear(320, 50)
        self.linear2 = nn.Linear(50, 26)
        
    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x),2)))
        x = x.view(-1 ,320)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.linear2(x))
        #no retorno el loss porque se lo hago con la magia
            
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
network = Red()
optimizer = optim.Adam(network.parameters())
def train(network, optimizer, train_loader, epoch):
    
    network.train()
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output = network(data)
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), '/results/model.pth')
            torch.save(optimizer.state_dict(), '/results/optimizer.pth')
def test(network, optimizer, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():    
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
