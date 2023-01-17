import torch 

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

%matplotlib inline
# Hyper parameters

num_epochs = 8

num_classes = 10

batch_size = 128

learning_rate = 0.002

IMG_SIZE = 28



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
train_set = pd.concat([pd.read_csv('../input/Kannada-MNIST/train.csv'), 

                       pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')], axis=0)

test_set = pd.read_csv('../input/Kannada-MNIST/test.csv')

sub = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')



train, val = train_test_split(train_set, test_size=0.1)

train.head()
len(train), len(val)
class MyDataset(Dataset):

    def __init__(self, images, labels, transform=None):

        super().__init__()

        self.X = images

        self.y = labels

        self.transform = transform

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        image = self.X.iloc[index, :]

        image = np.array(image).astype(np.uint8).reshape(28, 28, 1)

        if self.transform is not None:

            image = self.transform(image)

        

        label = self.y.iloc[index]

        return image, label
trans_train = transforms.Compose([transforms.ToPILImage(),

                                  transforms.RandomCrop(IMG_SIZE),

                                 transforms.RandomRotation(20),

                                 transforms.ToTensor()])

trans_valid = transforms.Compose([transforms.ToPILImage(),

                                 transforms.ToTensor()])

dataset_train = MyDataset(train.iloc[:, 1:], train.iloc[:, 0], trans_train)

dataset_valid = MyDataset(val.iloc[:, 1:], val.iloc[:, 0], trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 7 * 7, 10)

    

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #(1, 14, 14)

        x = self.pool(F.relu(self.bn2(self.conv2(x)))) #(1, 7, 7)

        # print('x.shape={}'.format(x.shape))

        x = x.view(-1, 64 * 7 * 7)

        x = self.fc(x)

        return x
model = SimpleCNN().to(device)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)



total_step = len(loader_train)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(loader_train):

        images = images.to(device)

        labels = labels.to(device)

        

        # Forward pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (i+1) % 100 == 0:

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 

                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model.eval()  

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in loader_valid:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

          

    print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))



# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')
dataset_valid = MyDataset(test_set.iloc[:,1:] , sub.iloc[:,1], transform=trans_valid)

loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)
model.eval()



preds = []

for batch_i, (data, target) in enumerate(loader_test):

    data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output.argmax(dim=1).cpu().numpy()

    for i in pr:

        preds.append(i)

sub.shape, len(preds)

sub['label'] = preds

sub.to_csv('s.csv', index=False)

sub