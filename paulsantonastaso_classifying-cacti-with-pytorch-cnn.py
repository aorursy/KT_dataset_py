import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from PIL import Image



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import torchvision

import torchvision.transforms as transforms



from sklearn.model_selection import train_test_split



import os, random

print(os.listdir("../input"))
# define data directories 

data_dir = '../input'

train_dir = data_dir + '/train/train/'

test_dir = data_dir + '/test/test/'
# visualize the label map & number of classes

labels = pd.read_csv("../input/train.csv")

labels.head()
# plot 10 random images from the training dataset

fig = plt.figure(figsize=(25, 8))

train_imgs = os.listdir("../input/train/train")

for idx, img in enumerate(np.random.choice(train_imgs, 10)):

    ax = fig.add_subplot(2, 10//2, idx+1, xticks=[], yticks=[])

    im = mpimg.imread("../input/train/train/" + img)

    plt.imshow(im, cmap="hot")

    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]

    ax.set_title(f'Label: {lab}')
# plot 6 random images with a heatmap from the training dataset

fig = plt.figure(figsize=(10, 10))

train_imgs = os.listdir("../input/train/train")

for idx, img in enumerate(np.random.choice(train_imgs, 6)):

    ax = fig.add_subplot(3, 6//2, idx+1, xticks=[], yticks=[])

    im = mpimg.imread("../input/train/train/" + img)

    lum_img = im[:,:,0]

    plt.imshow(lum_img, cmap="hot")

    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]

    ax.set_title(f'Label: {lab}')
# check the balance of the data labels

balance = labels['has_cactus'].value_counts()

print(balance)
# lets split the data to create a validation set using train_test_split() 

train, valid = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
# this gives us a pretty close 50, 25 , 25 % data split

print('Number of training samples:{0}'.format(len(train)))

print('Number of validation samples:{0}'.format(len(valid)))

print('Number of testing samples:{0}'.format(len(os.listdir(test_dir))))
# define hyper-params

num_epochs = 25

num_classes = 2

batch_size = 128

learning_rate = 0.002

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ole reliable - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

class CactiImageData(Dataset):

    def __init__(self, split_data, data_root = './', transform=None):

        super().__init__()

        self.df = split_data.values

        self.data_root = data_root

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_root, img_name)

        image = mpimg.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
# define some light data augmentation on the training set only

mean = [0.5, 0.5, 0.5]

std = [0.5, 0.5, 0.5]



train_transf = transforms.Compose([transforms.ToPILImage(),

                                   transforms.ToTensor()])



valid_transf = transforms.Compose([transforms.ToPILImage(),

                                  transforms.ToTensor()])
train_data = CactiImageData(split_data = train, data_root = train_dir, transform = train_transf)

valid_data = CactiImageData(split_data = valid, data_root = train_dir, transform = valid_transf)



train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers = 0)

valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size//2, shuffle = False, num_workers = 0)
# define our model architecture 

class CactiCNN(nn.Module):

    def __init__(self):

        super(CactiCNN, self).__init__()

        self.conv1 = nn.Sequential(

                nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0),

                nn.BatchNorm2d(32),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv2 = nn.Sequential(

                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1),

                nn.BatchNorm2d(64),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv3 = nn.Sequential(

                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),

                nn.BatchNorm2d(128),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv4 = nn.Sequential(

                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),

                nn.BatchNorm2d(256),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        

        self.fc = nn.Sequential(

                nn.Linear(256 * 2 * 2, 1024),

                nn.ReLU(inplace=True),

                nn.Dropout(0.25),

                nn.Linear(1024, 2))



    def forward(self,x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = x.view(x.shape[0],-1)

        x = self.fc(x)

        return x
# define our loss function and optimizer

model = CactiCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# train the model - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

total_step = len(train_loader)



for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

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
# test the models accuracy against the valid dataset

model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in valid_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

          

    print('Test Accuracy: {} %'.format(100 * correct / total))
# create a dataset and data loader for the test data

submit = pd.read_csv('../input/sample_submission.csv')

test_data = CactiImageData(split_data = submit, data_root = test_dir, transform = valid_transf)

test_loader = DataLoader(dataset = test_data, batch_size=32, shuffle=False, num_workers=0)
# create a submission file using the data in our test_loader

model.eval()

predict = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.to(device), target.to(device)

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        predict.append(i)



submit['has_cactus'] = predict

submit.to_csv('submission.csv', index=False)