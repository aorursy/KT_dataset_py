# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from random import randrange

import os



image_file=[]



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        i=randrange(10)

        if(i%2==0):

            image_file.append(dirname+filename)



# Any results you write to the current directory are saved as output.
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
import cv2 

img = cv2.imread('/kaggle/input/chinese-zodiac-signs/signs/test/rooster/00000165.jpg')



plt.imshow(img)

plt.show()
!pip install nonechucks
import nonechucks as nc



data_dir="/kaggle/input/chinese-zodiac-signs/signs"

batch_size = 16





train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(255),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.5, 0.5, 0.5],

                                                           [0.5, 0.5, 0.5])])



test_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.5, 0.5, 0.5],

                                                           [0.5, 0.5, 0.5])])



# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

valid_data = datasets.ImageFolder(data_dir + '/valid',transform=test_transforms)

test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)





train_data = nc.SafeDataset(train_data)

test_data = nc.SafeDataset(test_data)

valid_data = nc.SafeDataset(valid_data)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)



loaders_transfer={"train":trainloader,"test":testloader,"valid":validloader}
from collections import OrderedDict

use_cuda = torch.cuda.is_available()

model_transfer = models.resnet50(pretrained=True)



for param in model_transfer.parameters():

    param.requires_grad = False



od = OrderedDict() 

od['l1'] = nn.Linear(2048, 256)

od['a1'] = nn.ReLU()

od['d1'] = nn.Dropout(0.35)

od['l2'] = nn.Linear(256, 64)

od['a2'] = nn.ReLU()

od['d2'] = nn.Dropout(0.35)

od['l3'] = nn.Linear(64, 12)

od['softmax'] = nn.LogSoftmax(dim=1)



new_classifier = nn.Sequential(od)



for param in new_classifier.parameters():

  param.requires_grad=True



model_transfer.fc = new_classifier

fc_parameters = model_transfer.fc.parameters()





if use_cuda:

    model_transfer = model_transfer.cuda()

print(model_transfer)
import torch.optim as optim



criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.001)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np



def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    """returns trained model"""

    # initialize tracker for minimum validation loss

    valid_loss_min = np.Inf

    

    for epoch in range(1, n_epochs+1):

        # initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        

        ###################

        # train the model #

        ###################

        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()



            # initialize weights to zero

            optimizer.zero_grad()

            

            output = model(data)

            

            # calculate loss

            loss = criterion(output, target)

            

            # back prop

            loss.backward()

            

            # grad

            optimizer.step()

            

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            

            if batch_idx % 100 == 0:

                print('Epoch %d, Batch %d loss: %.6f' %

                  (epoch, batch_idx + 1, train_loss))

        

        ######################    

        # validate the model #

        ######################

        model.eval()

        for batch_idx, (data, target) in enumerate(loaders['valid']):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            output = model(data)

            loss = criterion(output, target)

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))



            

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        ## TODO: save the model if validation loss has decreased

        if valid_loss < valid_loss_min:

            torch.save(model.state_dict(), save_path)

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            valid_loss_min = valid_loss

            

    # return trained model

    return model



# train the model



model_transfer =  train(50, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')



# load the model that got the best validation accuracy (uncomment the line below)

model_transfer.load_state_dict(torch.load('model_transfer.pt'))
def test(loaders, model, criterion, use_cuda):



    # monitor test loss and accuracy

    test_loss = 0.

    correct = 0.

    total = 0.



    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):

        # move to GPU

        if use_cuda:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # update average test loss 

        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class

        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label

        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

        total += data.size(0)

            

    print('Test Loss: {:.6f}\n'.format(test_loss))



    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (

        100. * correct / total, correct, total))
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
class_names = ['dog', 'dragon', 'goat', 'horse', 'monkey', 'ox', 'pig', 'rabbit', 'rat', 'rooster', 'snake', 'tiger']



def predict_animal(img_path):

    # load the image and return the predicted breed

    img = load_input_image(img_path)

    model = model_transfer.cpu()

    model.eval()

    idx = torch.argmax(model(img))

    return class_names[idx]
from PIL import Image

import torchvision.transforms as transforms



def load_input_image(img_path):    

    image = Image.open(img_path).convert('RGB')

    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),

                                     transforms.ToTensor()])



    # discard the transparent, alpha channel (that's the :3) and add the batch dimension

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)

    return image
def run_app(img_path):

    img = Image.open(img_path)

    plt.imshow(img)

    plt.show()

    prediction = predict_animal(img_path)

    print("It looks like a {0}".format(prediction)) 
run_app('/kaggle/input/chinese-zodiac-signs/signs/train/ratt/00000650.jpg')

run_app('/kaggle/input/chinese-zodiac-signs/signs/test/dragon/00000013.jpg')