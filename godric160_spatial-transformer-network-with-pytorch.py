from __future__ import print_function



import torch

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader



import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
file_path = '../input/digit-recognizer'

train_df = pd.read_csv(os.path.join(file_path,'train.csv'))

test_df = pd.read_csv(os.path.join(file_path,'test.csv'))



print('Number of training samples: {0}'.format(len(train_df)))

print('Number of test samples: {0}'.format(len(test_df)))

print('Training sample columns: {0}'.format(train_df.columns.values))

print('Test sample columns: {0}'.format(test_df.columns.values))
print('Number of classes: {0}'.format(len(set(train_df['label']))))



plt.rcParams['figure.figsize'] = (8, 5)

plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())

plt.xticks(np.arange(len(set(train_df['label']))))

plt.xlabel('Class', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.grid('on', axis='y')
class mnist(Dataset):

    def __init__(self, file_path, is_train=True, transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])):

        

        self.transform = transform



        if is_train:

            train_path = os.path.join(file_path, 'train.csv')

            train = pd.read_csv(train_path)

            self.X = train.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(train.iloc[:,0].values)

        else:

            test_path = os.path.join(file_path, 'test.csv')

            test = pd.read_csv(test_path)

            self.X = test.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

    

    def __len__(self):

        return len(self.X)

        

    def __getitem__(self, index):

        if self.y is not None:

            return self.transform(self.X[index]), self.y[index]

        else:

            return self.transform(self.X[index])
import torch.nn.functional as F

import torch.nn as nn



class cnn(nn.Module):

    def __init__(self):

        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)

        self.fc2 = nn.Linear(50, 10)



        # Spatial transformer localization-network

        self.localization = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=7),

            nn.MaxPool2d(2, stride=2),

            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),

            nn.MaxPool2d(2, stride=2),

            nn.ReLU(True)

        )



        # Regressor for the 3 * 2 affine matrix

        self.fc_loc = nn.Sequential(

            nn.Linear(10 * 3 * 3, 32),

            nn.ReLU(True),

            nn.Linear(32, 3 * 2)

        )



        # Initialize the weights/bias with identity transformation

        self.fc_loc[2].weight.data.zero_()

        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))



    # Spatial transformer network forward function

    def stn(self, x):

        xs = self.localization(x)

        xs = xs.view(-1, 10 * 3 * 3)

        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 3)



        grid = F.affine_grid(theta, x.size())

        x = F.grid_sample(x, grid)



        return x



    def forward(self, x):

        # transform the input

        x = self.stn(x)



        # Perform the usual forward pass

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



batch_size = 64



train_dataset = mnist(file_path, is_train=True)

test_dataset = mnist(file_path,is_train=False)



train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
import torch.optim as optim



model = cnn().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

criterion = criterion.to(device)



def train(epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)



        optimizer.zero_grad()

        output = model(data)

        # loss = F.nll_loss(output, target)

        loss = criterion(output, target)



        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader), loss.item()))
def evaluate():

    with torch.no_grad():

        model.eval()

        test_loss = 0

        correct = 0

        for data, target in train_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)



            # sum up batch loss

            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # get the index of the max log-probability

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()



        test_loss /= len(train_loader.dataset)

        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'

              .format(test_loss, correct, len(train_loader.dataset),

                      100. * correct / len(train_loader.dataset)))
for epoch in range(1, 10):

    train(epoch)

    evaluate()
def prediciton():

    with torch.no_grad():

        model.eval()

        test_pred = torch.LongTensor()

        for i, data in enumerate(test_loader):

            data = data.to(device)

            output = model(data)



            pred = output.cpu().data.max(1, keepdim=True)[1]

            test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred
test_pred = prediciton()

out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], test_pred.numpy()],columns=['ImageId', 'Label'])

out_df.head()
out_df.to_csv('submission.csv', index=False)