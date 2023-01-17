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
!pip install torchsummary

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy
data_transforms = {

    'training_set': transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test_set': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
data_dir = '/kaggle/input/dogs-cats-images/dataset/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                          data_transforms[x]) for x in ['training_set', 'test_set']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[y], batch_size=16,

                                             shuffle=True, num_workers=4)

              for x in ['train', 'val'] for y in ['training_set', 'test_set']}

dataset_sizes = {x: len(image_datasets[y]) for x in ['train', 'val'] for y in ['training_set', 'test_set']}

class_names = image_datasets['training_set'].classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

%matplotlib inline



def imshow(img):

    img = img / 2 + 0.5  

    plt.imshow(np.transpose(img, (1, 2, 0)))



dataiter = iter(dataloaders['train'])

images, labels = dataiter.next()

print(images.shape,labels.shape)

images = images.numpy() 

fig = plt.figure(figsize=(25, 16))



for idx in np.arange(10):

    ax = fig.add_subplot(4, 10/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title(class_names[int(labels[idx])],fontsize=20,color='white')
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()



    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)



            for j in range(inputs.size()[0]):

                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)

                ax.axis('off')

                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                imshow(inputs.cpu().data[j])



                if images_so_far == num_images:

                    model.train(mode=was_training)

                    return

        model.train(mode=was_training)
model_ft = models.resnet18(pretrained=True)

for params in model_ft.parameters():

    params.requires_grad = False

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, len(class_names))



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=10)

visualize_model(model_ft)
import torch.nn as nn

import torch.nn.functional as F

class Net(nn.Module):

    

    def __init__(self):

        super(Net, self).__init__()

        

        self.block = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.Dropout(p=0.2)

        )

        self.block2 = nn.Sequential(

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.Dropout(p=0.2)

        )

        

            

        self.fc1 = nn.Linear(32*12*12, 512)

        self.batch_norm1 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 2)

        self.flatten = nn.Flatten()

        

    def forward(self, x):

        out = self.block(x)

        out = self.block2(out)

        out = self.block2(out)

        out = self.block2(out)

        out = self.flatten(out)

# flatten out a input for Dense Layer

        out = self.fc1(out)

        out = F.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        

        return out
from torchsummary import summary

model_ft = Net()

model_ft = model_ft.to(device)

print(summary(model_ft,(3,224,224)))





criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=10)

visualize_model(model_ft)