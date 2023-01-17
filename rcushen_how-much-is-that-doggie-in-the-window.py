import numpy as np

import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.utils.data import random_split, DataLoader



import os

import glob

import re

from random import shuffle

from PIL import Image



from tqdm import tqdm



%matplotlib inline



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device: {}'.format(device))
image_dir = "../input/images/Images"

image_paths = [path for path in glob.iglob(image_dir + '/*/*.jpg', recursive=True)]

pattern = re.compile("(?<=-)\w+(?=/n)")

breeds = [pattern.findall(image)[0] for image in image_paths]

dogs = [(image_paths[i], breeds[i]) for i in range(len(image_paths))]

shuffle(dogs)



num_sample_dogs = 6



test_ims = [Image.open(dogs[i][0]) for i in range(num_sample_dogs)]

test_ims_arrays = [np.array(test_im) for test_im in test_ims]



fig, ax = plt.subplots(ncols=num_sample_dogs, figsize=(18,12))

for dog in range(num_sample_dogs):

    ax[dog].imshow(test_ims_arrays[dog])

    ax[dog].get_xaxis().set_visible(False)

    ax[dog].get_yaxis().set_visible(False)

    ax[dog].set_title(dogs[dog][1]);
data_transforms = transforms.Compose([

    transforms.Resize(140),

    transforms.CenterCrop(128),

    transforms.ToTensor()

])



dataset = datasets.ImageFolder(image_dir, transform=data_transforms)

train_dataset, test_dataset = random_split(dataset, 

                                           (int(0.9*len(dataset)), 

                                            len(dataset)-int(0.9*len(dataset))))

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

print("Training set size: {}".format(len(train_dataset)))

print("Testing set size: {}".format(len(test_dataset)))
class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, 

                               kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, 

                               kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(12)

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 128 -> 64

        

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, 

                               kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, 

                               kernel_size=3, stride=1, padding=1)

        self.bn4 = nn.BatchNorm2d(48)

        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 64 -> 32

        

        self.conv5 = nn.Conv2d(in_channels=48, out_channels=96, 

                               kernel_size=3, stride=1, padding=1)

        self.bn5 = nn.BatchNorm2d(96)

        self.conv6 = nn.Conv2d(in_channels=96, out_channels=182, 

                               kernel_size=3, stride=1, padding=1)

        self.bn6 = nn.BatchNorm2d(182)

        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32 -> 16

        

        self.conv7 = nn.Conv2d(in_channels=182, out_channels=182, 

                               kernel_size=3, stride=1, padding=1)

        self.bn7 = nn.BatchNorm2d(182)

        self.conv8 = nn.Conv2d(in_channels=182, out_channels=256, 

                               kernel_size=3, stride=1, padding=1)

        self.bn8 = nn.BatchNorm2d(256)

        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 16 -> 8

        

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, 

                               kernel_size=3, stride=1, padding=1)

        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, 

                               kernel_size=3, stride=1, padding=1)

        self.bn10 = nn.BatchNorm2d(256)

        

        self.fc1 = nn.Linear(256*8*8, 10000)

        self.fc2 = nn.Linear(10000, 5000)

        self.fc3 = nn.Linear(5000, 1000)

        self.fc4 = nn.Linear(1000, 120)



    def forward(self, x):

        x = self.mp1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))

        x = self.mp2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))

        x = self.mp3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))

        x = self.mp4(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x)))))))

        x = F.relu(self.bn10(self.conv10(F.relu(self.bn9(self.conv9(x))))))

        

        x = x.view(x.shape[0], -1)

        

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        

        return x
convnet = ConvNet()
convnet.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(convnet.parameters(), lr=0.03)



epochs = 70

train_losses, test_losses = [], []

test_accuracies = []



for e in tqdm(range(epochs)):

    convnet.train()

    for ii, (images, labels) in enumerate(train_dataloader):

        images, labels = images.to(device), labels.to(device)



        optimizer.zero_grad()



        logits = convnet(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()



        train_losses.append(loss.item() / images.shape[0])

    

    test_loss = 0

    accuracy = 0

    convnet.eval()

    for ii, (images, labels) in enumerate(test_dataloader):

        images, labels = images.to(device), labels.to(device)

        

        with torch.no_grad():

            logits = convnet(images)

            loss = criterion(logits, labels)

            test_loss += loss.item() / images.shape[0]

            

            preds = logits.argmax(dim=1)

            accuracy += torch.mean((preds == labels).type(torch.FloatTensor))

            

    test_losses.append(test_loss / len(test_dataloader))

    test_accuracies.append(accuracy / len(test_dataloader))
train_losses_averaged = []

interval = 250

for i in range(0,len(train_losses),interval):

        try:

            train_losses_averaged.append(np.mean(train_losses[i:i+interval]))

        except IndexError:

            train_losses_averaged.append(np.mean(train_losses[i:]))
fig = plt.figure(figsize=(18, 12))

grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.2)

train_loss_ax = fig.add_subplot(grid[0,:])

test_loss_ax = fig.add_subplot(grid[1, 0])

test_acc_ax = fig.add_subplot(grid[1, 1])



train_loss_ax.plot(train_losses_averaged)

train_loss_ax.set_title('Training loss (average per 250 batches)')

train_loss_ax.set_xlabel('Batch')

test_loss_ax.plot(test_losses)

test_loss_ax.set_title('Test loss (average per epoch)')

test_loss_ax.set_xlabel('Epoch')

test_acc_ax.plot(test_accuracies, color='red')

test_acc_ax.set_title('Test accuracy (average per epoch)')

test_acc_ax.set_xlabel('Epoch')



plt.show()