# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import time



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
import torch

from torchvision import datasets

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LeNet(nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),

            nn.ReLU(),

            nn.MaxPool2d(2,2),

            nn.Conv2d(6, 16, 5),

            nn.ReLU(),

            nn.MaxPool2d(2,2)

        )

        self.fc = nn.Sequential(

            nn.Linear(in_features=16*4*4, out_features=120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.Sigmoid(),

            nn.Linear(84,10)

        )

    def forward(self, img):

        feature =- self.conv(img)

        output = self.fc(feature.view(img.shape[0], -1))

        return output

    

net = LeNet()

print(net)
transform = transforms.ToTensor()



x_train, x_val, y_train, y_val = train_test_split(

    train.values[:,1:], train.values[:,0], test_size=0.2)



batch_size = 128



train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train.reshape(-1,1, 28,28).astype(np.float32)/255),

                                               torch.from_numpy(y_train))



val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val.reshape(-1,1, 28,28).astype(np.float32)/255),

                                               torch.from_numpy(y_val))



test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test.values[:,:].reshape(-1,1, 28,28).astype(np.float32)/255))



# data loader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)



lr, n_epochs = 0.001, 30

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
n_epochs = 30

for epoch in range(n_epochs):

    start = time.time()

    train_loss = 0.0

    for data, target in train_loader:

        optimizer.zero_grad()

        output = net(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*data.size(0)

    

    train_loss = train_loss/len(train_loader.dataset)

    

    print(f"Epoch: {epoch}, train loss: {train_loss}, elapsed {(time.time() - start)}")
val_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



net.eval() # prep net for evaluation



for data, target in val_loader:

    # forward pass: compute predicted outputs by passing inputs to the net

    output = net(data)

    # calculate the loss

    loss = criterion(output, target)

    # update val loss 

    val_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)

    # compare predictions to true label

    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    # calculate val accuracy for each object class

    for i in range(len(target)):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# calculate and print avg val loss

val_loss = val_loss/len(val_loader.sampler)

print('val Loss: {:.6f}\n'.format(val_loss))



for i in range(10):

    if class_total[i] > 0:

        print('val Accuracy of %5s: %2d%% (%2d/%2d)' % (

            str(i), 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('val Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nval Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
net.eval() # prep net for evaluation



preds = []



for data in test_loader:

    # forward pass: compute predicted outputs by passing inputs to the net

    output = net(data[0])

    # calculate the loss

    _, pred = torch.max(output, 1)

    preds.extend(pred.tolist())

    # compare predictions to true label

submission['Label'] = preds

submission.to_csv('submission.csv', index=False)