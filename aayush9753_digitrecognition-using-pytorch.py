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
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
#Getting the label column
train_labels = np.array(train['label'])
# m = No of Exaples
m_train = train.shape[0] #m in training data
m_test = test.shape[0]  #m in testing data
#reshaping the long 1D vector of shape 1*784 into a 3D vector of shape 1*28*28 
train_data = np.array(train.loc[:,'pixel0':]).reshape(m_train,1,28,28)
test_data = np.array(test.loc[:,'pixel0':]).reshape(m_test,1,28,28)
k = 0
while k<2:
    i = random.randint(0,42000)
    plt.imshow(train_data[i,0,:,:])
    plt.show()
    print(f"The label for the above image is {train_labels[i]}")
    k+=1
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
X_train.shape, X_val.shape, X_test.shape
import numbers
class RandomRotation(object):


    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):

        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):

        
        def rotate(img, angle, resample=False, expand=False, center=None):

                
            return img.rotate(angle, resample, expand, center)

        angle = self.get_params(self.degrees)

        return rotate(img, angle, self.resample, self.expand, self.center)
class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)
# transformations to be applied on images
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
class DigitDataset(Dataset):

    def __init__(self,images,labels,transfrom = transform):
        # Initialize data, download, etc.
        self.x_data = torch.from_numpy(images/255.) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(labels) # size [n_samples, 1]
        self.n_samples = images.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
train_dataset = DigitDataset(X_train,y_train,transforms.Compose([RandomRotation(degrees=20), RandomShift(3),
                                                                 transforms.Normalize(mean=(0.5,), std=(0.5,)),
                             transform]))
val_dataset = DigitDataset(X_val,y_val,transform)
test_dataset = DigitDataset(X_test,y_test,transform)
batch_size=64
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
#Checking the device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
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
            nn.Linear(64 * 7 * 7, 512),
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

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=50)
# getting predictions on test set and measuring the performance
correct_count, all_count = 0, 0
for images,labels in test_loader:
  for i in range(len(labels)):
    images = images.cuda()
    images = images.type(torch.cuda.FloatTensor)
    labels = labels.cuda()
    img = images[i].view(1, 1, 28, 28)
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
test = test_data/255 #Normalizing the data
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
print(prediction[:5])
k = np.arange(1,28001)
submission = pd.DataFrame({
        "ImageId":k ,
        "Label": prediction

    })

submission.to_csv('Digit_Recognition_submission.csv', index=False)
