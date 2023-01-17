# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform

from torchvision import models



# adapted from https://www.kaggle.com/dansbecker/programming-in-tensorflow-and-keras



import json

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



from PIL import Image



%matplotlib inline 
# Exploring train.csv file

df = pd.read_csv('../input/train.csv')

df.head()
#Dataset class



class ImageDataset(Dataset):

    



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with labels.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.data_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         # getting path of image

        image = Image.open(img_name).convert('RGB')                                # reading image and converting to rgb if it is grayscale

        label = np.array(self.data_frame['Category'][idx])                         # reading label of the image

        

        if self.transform:            

            image = self.transform(image)                                          # applying transforms, if any

        

        sample = (image, label)        

        return sample
# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor

transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.ToTensor()                               

                                ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)     #Training Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice
#Checking training sample size and label

for i in range(len(trainset)):

    sample = trainset[i]

    print(i, sample[0].size(), " | Label: ", sample[1])

    if i == 9:

        break
# Visualizing some sample data

# obtain one batch of training images

dataiter = iter(trainloader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):                                             #Change the range according to your batch-size

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
model = models.resnext101_32x8d(pretrained=True)
for param in model.parameters():

    param.requires_grad = True

    

for param in model.parameters():

    param.requires_grad = True

    

    

for name, child in model.named_children():

    if name in ['layer3', 'layer4', 'fc']:

        print(name + ' is unfrozen')

        for param in child.parameters():

            param.requires_grad = True

    else:

        print(name + ' is frozen')

        for param in child.parameters():

            param.requires_grad = False
import torch.nn as nn



# Add on classifier

model.fc = nn.Sequential(

                      nn.Linear(2048, 4096), 

                      nn.ReLU(), 

                      nn.Linear(4096, 256), 

                      nn.ReLU(),

                      nn.Linear(256, 67),                   

                      nn.LogSoftmax(dim=1))
model.fc


if train_on_gpu:

    model.cuda()




# Loss function to be used

criterion = nn.CrossEntropyLoss()   # You can change this if needed



# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9)













epochs=20



for e in range(epochs):

    running_loss = 0

    #corr = torch.long(0)

    corr = torch.tensor((0), dtype=torch.int64)    

    for images, labels in trainloader:

        # Flatten images 

        

        images = images.cuda()

        labels=labels.cuda()

        # Training pass

        optimizer.zero_grad()

        

        output = model(images)

        _, out = torch.max(output, 1)

        loss = criterion(output, labels)

        #print(out,labels)

        #This is where the model learns by backpropagating

        loss.backward()

        corr+=sum(out==labels)

        #print(corr)

        #And optimizes its weights here

        optimizer.step()

        

        running_loss += loss.item()

    else:

        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

        print(corr,"/10934")





#Exit training mode and set model to evaluation mode

model.eval() # eval mode



# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
#Loading test data to make predictions



testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
predictions=[]

# iterate over test data to make predictions

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        





 

submission['Category'] = predictions  
#submission['Category'] = predictions             #Attaching predictions to submission file



#saving submission file

submission.to_csv('submission1.csv', index=False, encoding='utf-8')
