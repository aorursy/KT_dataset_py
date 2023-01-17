import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from torchvision.utils import make_grid



import math

import random



import numbers



import matplotlib.pyplot as plt

%matplotlib inline
random_seed = 7

random.seed(random_seed)

np.random.seed(random_seed)

torch.manual_seed(random_seed)

print('Random seed is now: {}'.format(random_seed))
train_df = pd.read_csv('../input/train.csv')



n_train = len(train_df)

n_pixels = len(train_df.columns) - 1

n_class = len(set(train_df['label']))



print('Number of training samples: {0}'.format(n_train))

print('Number of training pixels: {0}'.format(n_pixels))

print('Number of classes: {0}'.format(n_class))
test_df = pd.read_csv('../input/test.csv')



n_test = len(test_df)

n_pixels = len(test_df.columns)



print('Number of train samples: {0}'.format(n_test))

print('Number of test pixels: {0}'.format(n_pixels))
random_sel = np.random.randint(n_train, size=8)



grid = make_grid(torch.Tensor((train_df.iloc[random_sel, 1:].as_matrix()/255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)

plt.rcParams['figure.figsize'] = (16, 2) # units in Inchs

plt.imshow(grid.numpy().transpose((1,2,0)))

plt.axis('off')

print(*list(train_df.iloc[random_sel, 0].values), sep = ', ')
plt.rcParams['figure.figsize'] = (8, 5)

plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())

plt.xticks(np.arange(n_class))

plt.xlabel('Class', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.grid('on', axis='y')
class MNIST_data(Dataset):

    """MNIST data set"""

    

    def __init__(self, file_path):

        n_pixels = 28 * 28

        df = pd.read_csv(file_path)

        

        if len(df.columns) == n_pixels:

            # test data

            self.X = df.values.reshape((-1,1,28,28)).astype(np.float) / 255

            self.y = None

        else:

            # training data

            self.X = df.iloc[:,1:].values.reshape((-1,1,28,28)).astype(np.float) / 255

            self.y = torch.from_numpy(df.iloc[:,0].values)

        

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx): #x[2]

        if self.y is not None: #in case of training and test set

            return self.X[idx], self.y[idx] 

        else: # in case of submission datasets  

            return self.X[idx] 
from torch.utils.data import Subset



batch_size = 64



train_dataset_all = MNIST_data('../input/train.csv')



all_train_len = len(train_dataset_all)

indices = list(range(all_train_len))

validation_split = 0.2

split = int(np.floor(validation_split * all_train_len))

np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]



train_dataset = Subset(train_dataset_all, train_indices)

val_dataset = Subset(train_dataset_all, val_indices)



test_dataset = MNIST_data('../input/test.csv')





train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,

                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                           batch_size=batch_size, shuffle=False)
class LeNet(nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1   = nn.Linear(256, 120)

        self.fc2   = nn.Linear(120, 84)

        self.fc3   = nn.Linear(84, 10)



    def forward(self, x):

        out = F.relu(self.conv1(x))

        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))

        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))

        out = F.relu(self.fc2(out))

        out = self.fc3(out)

        return out
model = LeNet()



optimizer = optim.Adam(model.parameters(), lr=0.003)



criterion = nn.CrossEntropyLoss()





if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
def train(epoch):

    model.train()

    #exp_lr_scheduler.step()



    for batch_idx, (data, target) in enumerate(train_loader):

        

        if torch.cuda.is_available():

            data = data.float().cuda()

            target = target.cuda()

        

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        

        loss.backward()

        optimizer.step()

        

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))
def evaluate(data_loader):

    model.eval()

    loss = 0

    correct = 0

    

    for data, target in data_loader:

        if torch.cuda.is_available():

            data = data.float().cuda()

            target = target.cuda()

        

        output = model(data)

        



        pred = output.argmax(dim=1)

        correct += (pred == target).cpu().sum()

    loss /= len(data_loader.dataset)

        

    print('\nAccuracy: {}/{} ({:.3f}%)\n'.format(  

         correct, len(data_loader.dataset), 

        100. * correct / len(data_loader.dataset)))
n_epochs = 15



for epoch in range(n_epochs):

    train(epoch)

    print("train data:")

    evaluate(train_loader)

    print("val data:")

    evaluate(val_loader)
def prediciton(data_loader):

    model.eval()

    test_pred = torch.LongTensor()

    

    for i, data in enumerate(data_loader):

        data = data.float()

        if torch.cuda.is_available():

            data = data.cuda()

            

        output = model(data)

        

        pred = output.cpu().argmax(dim=1)

        test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred
test_pred = prediciton(test_loader)
out_df = pd.DataFrame({'ImageId':list(range(1, len(test_dataset) + 1)),'Label':test_pred.numpy()})
out_df.head()
out_df.to_csv('submission.csv', index=False)