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
!pip install albumentations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import random
import math
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from torch.optim import lr_scheduler, Adam, SGD

import torch.nn.functional as F
from torch.autograd import Variable

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import os
import copy

import random

import cv2

import albumentations as A
from IPython.display import Image

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
test_path = {}
for dirname, _, filenames in os.walk('/kaggle/input/mnist-but-chinese/MNIST_Chinese_Hackathon/Testing_Data/'):
    for filename in filenames:
        test_path[str(os.path.join(filename))] = str(os.path.join(dirname, filename))
        #test_path.append(str(os.path.join(dirname, filename)))
        #test_name.append(int(os.path.join(filename)))
        
train_path = {}
for dirname, _, filenames in os.walk('/kaggle/input/mnist-but-chinese/MNIST_Chinese_Hackathon/Training_Data/'):
    for filename in filenames:
        train_path[str(os.path.join(filename))] = str(os.path.join(dirname, filename))
len_test = len(test_path)
len_train = len(train_path)
test = pd.DataFrame()
train = pd.DataFrame()
name = []
path = []

for i in range(1,len_test+1):
    name.append(i)
    path.append(test_path[str(i)])
test['name'] = name
test['path'] = path
name = []
path = []

for i in range(1,len_train+1):
    name.append(i)
    path.append(train_path[str(i)])
train['name'] = name
train['path'] = path
train.head()
l = train.loc[2,'path']
l
train_images = []
k = np.array(train['path'])
for i in k:
    train_images.append(mpimg.imread(i).reshape(1,64,64))
train_images = np.array(train_images)

test_images = []
k = np.array(test['path'])
for i in k:
    test_images.append(mpimg.imread(i).reshape(1,64,64))
test_images = np.array(test_images)
train_images.shape, test_images.shape
train_labels = np.array(pd.read_csv("../input/mnist-but-chinese/MNIST_Chinese_Hackathon/train.csv")["code"])
count = []
for i in range(0,16):
    count.append(sum(train_labels==i))
count
pd.DataFrame(train_labels).nunique()
img = train_images[1].reshape(64,64)
imgplot = plt.imshow(img)
plt.show()
print(train_labels[1])
img = train_images[9997].reshape(64,64)
imgplot = plt.imshow(img)
plt.show()
print(train_labels[9997])
for i in range(len(train_labels)):
    if train_labels[i] == 15:
        train_labels[i] = 0
train_labels[:20]
train_images.shape, train_labels.shape, test_images.shape

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print(device)
class DigitDataset(Dataset):

    def __init__(self,images,labels,transfrom):
        # Initialize data, download, etc.
        self.x_data = torch.from_numpy(images) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(labels) # size [n_samples, 1]
        self.n_samples = images.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

aug_transform = A.Compose([
    A.RandomCrop(width=50, height=50),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transform = transforms.Compose([transforms.ToTensor(),
                              
                              ])
train_dataset = DigitDataset(X_train,y_train,transforms.Compose([aug_transform,
                                                                 transforms.Normalize(mean=(0.5,), std=(0.5,)),
                             transform]))
val_dataset = DigitDataset(X_val,y_val,transform)
test_dataset = DigitDataset(X_test,y_test,transform)
batch_size=128
# defining trainloader, valloader and testloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
# shape of training data
dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

# visualizing the training images
plt.imshow(images[0].numpy().squeeze(), cmap='gray')
# shape of validation data
dataiter = iter(val_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

# visualizing the training images
plt.imshow(images[0].numpy().squeeze(), cmap='gray')
print(labels[0])

64*64
class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
          
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
          
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 15),
        )
          

                

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x   

# defining the model
model = Net()
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)
# defining the loss function
criterion = nn.CrossEntropyLoss()
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)
dataset_sizes = {}
dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(val_dataset)
X_test.shape
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time() #Return the time in seconds since the epoch as a floating point number

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
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
model = model.to(device)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)
model.eval()
# getting predictions on test set and measuring the performance
correct_count, all_count = 0, 0
for images,labels in test_loader:
  for i in range(len(labels)):
    images = images.cuda()
    images = images.type(torch.cuda.FloatTensor)
    labels = labels.cuda()
    img = images[i].view(1, 1, 64, 64)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.cpu()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

test = test_images #Normalizing the data
test = torch.from_numpy(test)  # Converting into Tensors
test = test.type(torch.cuda.FloatTensor)
with torch.no_grad():
    outputs = model(test.cuda())
ps = torch.exp(outputs)

#max_value is the value of highest no. in each 10-dim vector 
#index is the index of that max value 
max_value, index = torch.max(ps,axis=1) 

index = index.cpu()
#Converting Prediction to numpy for Submission
prediction = index.numpy()

print(prediction.shape)
print(prediction[:10])
len(prediction)
for i in range(len(prediction)):
    if prediction[i] == 0:
        prediction[i]=15
prediction[:10]
k = np.arange(1,len(prediction)+1)
submission = pd.DataFrame({
        "id":k ,
        "code": prediction

    })

submission.to_csv('Digit_Recognition_submission.csv', index=False)
