#!pip install torchvision
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing

import os
import glob
import sys
sys.setrecursionlimit(100000)  #this will increase the capacity of the stack 

import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data import datasets
from torchvision import models

import imageio

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

%config IPCompleter.greedy=True
print(torch.__version__)
os.listdir('../input')
USE_GPU = True
traindata_dir = '../input/aptos2019-blindness-detection/train_images/'
train_label = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_label.head(10)
#print(traindata_dir)

#print(os.listdir(traindata_dir))

train_label["diagnosis"].value_counts().plot(kind="pie")
#train_label = pd.read_csv('../input/aptos2019-blindness-detection/train.csv', header=0).iloc[:,2:4]
#train_label.head(20)
testdata_dir = '../input/aptos2019-blindness-detection/test_images'
test_label = pd.read_csv("../input/aptos2019-blindness-detection/test.csv", encoding='latin-1')

#print(os.listdir(testdata_dir))
test_label.head()

plt.figure(figsize=[15,15])
i = 1
for img_name in train_label['id_code'][:12]:
    img = mpimg.imread(traindata_dir + img_name + '.png')
    plt.subplot(6,4,i)
    plt.imshow(img)
    i += 1
plt.show()
class DRTrainDataset(Dataset):
    def __init__(self, data_label, data_dir, transform):
        super().__init__()
        self.data_label = data_label
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_label)
    
    def __getitem__(self, index):       
        img_name = self.data_label.id_code[index] + '.png'
        label = self.data_label.diagnosis[index]          
        img_path = os.path.join(self.data_dir, img_name)   
            
        image = mpimg.imread(img_path)
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        
        image = self.transform(image)
        
        
        return image, label
from IPython.display import Image

listOfImageNames = ['../input/aptos2019-blindness-detection/train_images/000c1434d8d7.png',
                    '../input/aptos2019-blindness-detection/train_images/001639a390f0.png',
                   '../input/aptos2019-blindness-detection/train_images/002c21358ce6.png',
                    '..//input/aptos2019-blindness-detection/train_images/005b95c28852.png',
                    '../input/aptos2019-blindness-detection/train_images/005b95c28852.png'
                   ]

for imageName in listOfImageNames:
    display(Image(filename=imageName))

train_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.Resize(265),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'), 
                                  transforms.Resize(265),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = DRTrainDataset(data_label = train_label, data_dir = traindata_dir, transform = train_transform)
test_data = DRTrainDataset(data_label = test_label, data_dir = testdata_dir, transform = test_transform)


train_loader = DataLoader(dataset = train_data, batch_size=64, shuffle=True)

test_loader = DataLoader(dataset = test_data, batch_size=64)

import torch
from torchvision import datasets, transforms

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50()

model.load_state_dict(torch.load("../input/resnet50/resnet50.pth"))

model
# Build a feed-forward network
#I'll freeze the parameters so I don't back-propagate through them

for param in model.parameters():
    param.requires_grad = False
    
#print(model)
import time

model.fc = nn.Linear(2048, 5)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
# # number of epochs to train the model
# num_epochs = 20

# valid_loss_min = np.Inf # track change in validation loss

# for epoch in range(1, num_epochs+1):

#     # keep track of training and validation loss
#     train_loss = 0.0
    
    
#     ###################
#     # train the model #
#     ###################
#     model.train()
#     for inputs, label in train_loader:
#         # move tensors to GPU if CUDA is available
#         inputs, label = inputs.to(device), label.to(device)
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(inputs)
#         # calculate the batch loss
#         loss = loss_func(output, label)
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         # update training loss
#         train_loss += loss.item()*inputs.size(0)
    
    
#     # print training/validation statistics 
#     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss))
epochs = 15
steps = 0
running_loss = 0
print_every = 5

loss_log=[] 

for epoch in range(epochs):
    model.train()
        
    for ii, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
             
        optimizer.zero_grad()
        output = model(inputs)                    
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        
        if ii % 1000 == 0:
            loss_log.append(loss.item())
        # inside the for-loop:
        if epoch % 10 == 9:
          torch.save(model.state_dict(), 'train_valid_exp4-epoch{}.pth'.format(epoch+1))
    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))

      
initial_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
test_data = DRTrainDataset(data_label = initial_submission, data_dir = testdata_dir, transform = test_transform)
test_loader = DataLoader(dataset = test_data, shuffle=False)
%%time
# Prediction
predict = []
model.eval()

with torch.no_grad():
    for i, (inputs, _) in enumerate(test_loader):
        inputs = inputs.cuda()
        output = model(inputs)
        output = output.cpu().detach().numpy()
        predict.append(output[0])
initial_submission['diagnosis'] = np.argmax(predict, axis=1)
initial_submission.head(15)
initial_submission.to_csv('submission.csv', index=False)