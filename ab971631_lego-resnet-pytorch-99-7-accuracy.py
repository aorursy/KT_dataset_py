import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.autograd import Variable

from torch import optim

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, models, transforms

import time

import os

import copy

import torch.optim as optim

from torch.optim import lr_scheduler

from torchvision import transforms, utils

from PIL import Image

from tqdm import tqdm

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

import cv2   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test=pd.read_csv("/kaggle/input/lego-dataset/Test.csv")

train=pd.read_csv("/kaggle/input/lego-dataset/Train.csv")
train.head()
# loading training images

train_img = []

for img_name in tqdm(train['name']):

    # defining the image path

    image_path = '/kaggle/input/lego-dataset/train/' + str(img_name) 

    # reading the image

    img = cv2.imread(image_path,0)

    img = img / 255.0

    img=img.reshape(200,200,1)

    # converting the type of pixel to float 32

    img = img.astype('float32')

    # appending the image into the list

    train_img.append(img)



# converting the list to numpy array

train_x = np.array(train_img)

# defining the target

train_y = train['category'].values-1

train_x.shape
def display_examples(images, labels):

    """

        Display 25 images from the images array with its corresponding labels

    """

    

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of images of the dataset", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i].reshape(200,200), cmap=plt.cm.binary)

        plt.xlabel(labels[i])

    plt.show()
display_examples(train_x, train_y+1)
from sklearn.model_selection import train_test_split

train_xx, val_x, train_yy, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)

(train_xx.shape, train_yy.shape), (val_x.shape, val_y.shape)
class RandomRotation(object):

    """

    https://github.com/pytorch/vision/tree/master/torchvision/transforms

    Rotate the image by angle.

    Args:

        degrees (sequence or float or int): Range of degrees to select from.

            If degrees is a number instead of sequence like (min, max), the range of degrees

            will be (-degrees, +degrees).

        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):

            An optional resampling filter.

            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters

            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.

        expand (bool, optional): Optional expansion flag.

            If true, expands the output to make it large enough to hold the entire rotated image.

            If false or omitted, make the output image the same size as the input image.

            Note that the expand flag assumes rotation around the center and no translation.

        center (2-tuple, optional): Optional center of rotation.

            Origin is the upper left corner.

            Default is the center of the image.

    """



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

        """Get parameters for ``rotate`` for a random rotation.

        Returns:

            sequence: params to be passed to ``rotate`` for random rotation.

        """

        angle = np.random.uniform(degrees[0], degrees[1])



        return angle



    def __call__(self, img):

        """

            img (PIL Image): Image to be rotated.

        Returns:

            PIL Image: Rotated image.

        """

        

        def rotate(img, angle, resample=False, expand=False, center=None):

            """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)

            Args:

            img (PIL Image): PIL Image to be rotated.

            angle ({float, int}): In degrees degrees counter clockwise order.

            resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):

            An optional resampling filter.

            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters

            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.

            expand (bool, optional): Optional expansion flag.

            If true, expands the output image to make it large enough to hold the entire rotated image.

            If false or omitted, make the output image the same size as the input image.

            Note that the expand flag assumes rotation around the center and no translation.

            center (2-tuple, optional): Optional center of rotation.

            Origin is the upper left corner.

            Default is the center of the image.

            """

                

            return img.rotate(angle, resample, expand, center)



        angle = self.get_params(self.degrees)



        return rotate(img, angle, self.resample, self.expand, self.center)

    

class RandomShift(object):

    def __init__(self, shift):

        self.shift = shift

        

    @staticmethod

    def get_params(shift):

        """Get parameters for ``rotate`` for a random rotation.

        Returns:

            sequence: params to be passed to ``rotate`` for random rotation.

        """

        hshift, vshift = np.random.uniform(-shift, shift, size=2)



        return hshift, vshift 

    def __call__(self, img):

        hshift, vshift = self.get_params(self.shift)

        

        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)
import numbers

train_transform= transforms.Compose([

            transforms.ToPILImage(),

            RandomRotation(20),

            RandomShift(3),

            transforms.RandomHorizontalFlip(), # Horizontal Flip

#             transforms.RandomVerticalFlip(p=0.5),

#             transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),

            transforms.ToTensor(),  #Convereting the input to tensor

            ])



transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.ToTensor()])
class MyDataset(Dataset):

    

    #initialise the class variables - transform, data, target

    def __init__(self, data, target=None, transform=None): 

        self.transform = transform

        self.target=target

        if self.target is not None:

            self.data = (data) 

            self.target = torch.from_numpy(target).long() 

        else:

            self.data = (data)

    

    #retrieve the X and y index value and return it

    def __getitem__(self, index): 

        if self.target is not None: 

            return self.transform(self.data[index]), self.target[index]

        else:

            return self.transform(self.data[index])

    

    #returns the length of the data

    def __len__(self): 

        return len(list(self.data))

    

train_dataset = MyDataset(train_xx, train_yy,train_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)



val_dataset = MyDataset(val_x, val_y,transform)

test_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)





dataloaders = {'train' : train_loader, 'val' : test_loader }

dataset_sizes = {'train' : len(train_xx), 'val' : len(val_x) }
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0

            f1_batch=0



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

                f1_batch += f1_score(labels.data.cpu(),preds.cpu(),average='weighted')

       

           

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_f1=f1_batch / len(dataloaders[phase])

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f} F1_score: {:.4f}'.format(

                phase, epoch_loss, epoch_acc,epoch_f1))



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
model_ft = models.resnet18(pretrained=False)

model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)



num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 16)



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss().to(device)



# Observe that all parameters are being optimized

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.15)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=20)
test_img = []

for img_name in tqdm(test['name']):

    # defining the image path

    image_path = '/kaggle/input/lego-dataset/test/' + str(img_name) 

    # reading the image

    img = cv2.imread(image_path,0)

    img = img / 255.0

    img=img.reshape(200,200,1)

    # converting the type of pixel to float 32

    img = img.astype('float32')

    # appending the image into the list

    test_img.append(img)



# converting the list to numpy array

test_x = np.array(test_img)



test_x.shape
fig = plt.figure(figsize=(10,10))

fig.suptitle("Some examples of images of the dataset", fontsize=16)

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_x[i].reshape(200,200), cmap=plt.cm.binary)

#     plt.xlabel(labels[i])

plt.show()
val_dataset = MyDataset(data=test_x, transform=transform)

test_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
def prediciton(data_loader):

    model_ft.eval()

    test_pred = torch.LongTensor()

    

    for i, data in enumerate(data_loader):

        data = Variable(data, volatile=True)

        if torch.cuda.is_available():

            data = data.cuda()

            

        output = model_ft(data)

        

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred
test_pred = prediciton(test_loader)
test["category"]=test_pred.numpy()+1

test.head()
test.to_csv("sub.csv",header=True,index=False)