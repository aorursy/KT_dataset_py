# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.listdir('../input/color-flowers/color_data/Test/')
root = '../input/color-flowers/color_data'

#Test_path = '../input/rgb2gray'
import cv2
import matplotlib.pyplot as plt

import sys



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torch.autograd import Variable



from torchvision import datasets, transforms, models
import numpy as np

%matplotlib inline


transform = transforms.Compose([transforms.Resize(224),

                                transforms.CenterCrop(128),

                                #transforms.RandomHorizontalFlip(),

                                transforms.ToTensor()

                                #transforms.Normalize([0.5, 0.5, 0.5],

                                #                     [0.5, 0.5, 0.5])

                               ])



# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(root + '/Train', transform=transform)

test_data = datasets.ImageFolder(root + '/Test', transform=transform)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def show_img(orig, gray, conv_gray):

    orig = np.transpose(orig, (1, 2, 0))

    #gray = np.transpose(gray, (1, 2, 0))

    #gray_img = transforms.ToPILImage()

    #gray = transforms.functional.to_grayscale(gray_img(np.uint8(orig)), num_output_channels=3)

    conv_gray = np.transpose(conv_gray, (1, 2, 0))

    fig=plt.figure(figsize=[10,5])

    

    #orig = orig.swapaxes(0, 1).swapaxes(1, 2)

    #gray = gray.swapaxes(0, 1).swapaxes(1, 2)

    #converted_gray = conv_gray.swapaxes(0, 1).swapaxes(1, 2)

    

    # Normalize for display purpose

    orig     = (orig - orig.min()) / (orig.max() - orig.min())

    gray    = (gray - gray.min()) / (gray.max() - gray.min())

    conv_gray = (conv_gray - conv_gray.min()) / (conv_gray.max() - conv_gray.min())

    

    fig.add_subplot(1, 3, 1, title='Original')

    plt.imshow(orig)

    

    fig.add_subplot(1, 3, 2, title='Grey')

    plt.imshow(gray,cmap='gray')

    

    fig.add_subplot(1, 3, 3, title='Converted color')

    plt.imshow(conv_gray)

    

    fig.subplots_adjust(wspace = 0.5)

    plt.show()

    

# To test

# show_img(cifar10_train[0][0].numpy(), cifar10_train[1][0].numpy(), cifar10_train[2][0].numpy())
class RGB2GRAY(nn.Module):

    

    def __init__(self):

    

        super(RGB2GRAY, self).__init__()

                                                            # 128 x 128 x 3 (input)



        self.conv1e = nn.Conv2d(3, 128, 3)

        self.conv2e = nn.Conv2d(128,256,3)

        self.conv3e = nn.Conv2d(256,512,3)

        self.mp1e   = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  



        self.mp1d = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv1d = nn.ConvTranspose2d(512, 256, 3)

        self.conv2d = nn.ConvTranspose2d(256, 128, 3)

        self.conv3d = nn.ConvTranspose2d(128, 3, 3)

       

    

    def forward(self, x):

        # Encoder

        x = self.conv1e(x)

        x = F.relu(x)

        x = self.conv2e(x)

        x = F.relu(x)

        x = self.conv3e(x)

        x = F.relu(x)

        x, i = self.mp1e(x)

        

         # Decoder

        x = self.mp1d(x, i)

        x = self.conv1d(x)

        x = F.relu(x)

        x = self.conv2d(x)

        x = F.relu(x)

        x = self.conv3d(x)

        

        return x
autoencoder = RGB2GRAY()

criterion = nn.MSELoss()

parameters = list(autoencoder.parameters())

optimizer = optim.Adam(parameters, lr=0.005)
autoencoder.to(device)
train_loss = []

valid_loss = []



# Training the model 



epochs = 30

#steps = 0





for e in range(epochs):

    running_loss = 0

    running_iter = 0

    for images,labels in trainloader:

        #images_gray = cv2.cvtColor(cv2.UMat(images), cv2.COLOR_BGR2GRAY)

        PILimage = transforms.ToPILImage()

        img = PILimage(images[0])

        images_gray = transforms.functional.to_grayscale(img, num_output_channels=3)

        

        image = Variable(images).to(device)

        tensor = transforms.ToTensor()

        image_gray = Variable(tensor(images_gray)).to(device)

        image_gray = image_gray.view(-1,3,128,128)

        #print(image.shape)

        #print(image_gray.shape)

        optimizer.zero_grad()

    

        output = autoencoder(image) 

        #print(output.shape)

        loss = criterion(output, image_gray)           # remove [1:] and test

        loss.backward()              # For gradient calcultion

        optimizer.step()             # Optimizng - Tuning the weights of the model

    

        running_iter +=1

        running_loss += loss.item()

    

    

    autoencoder.eval()

    test_loss = 0

    test_iter = 0

    

    with torch.no_grad():

        for images, labels in testloader:

            #images_gray = cv2.cvtColor(cv2.UMat(images), cv2.COLOR_BGR2GRAY)

            PILimage = transforms.ToPILImage()

            img = PILimage(images[0])

            images_gray = transforms.functional.to_grayscale(img, num_output_channels=3)

            tensor = transforms.ToTensor()        

            

            image = Variable(images).to(device)

            image_gray = Variable(tensor(images_gray)).to(device)

            img_gray = image_gray.view(-1,3,128,128)

            

            output = autoencoder(image)

            loss = criterion(output, img_gray)

        

            test_iter +=1

            test_loss += loss.item()

      

    

    # Let's visualize the first image of the last batch in our validation set

    orig = image[0].cpu()

    gray = image_gray[0].cpu()

    conv_gray = output[0].cpu()

    

    orig = orig.data.numpy()

    gray = gray.data.numpy()

    conv_gray = conv_gray.data.numpy()

    #print(orig.shape)

    #print(gray.shape)

    #print(conv_gray.shape)

    print("Epoch:",e+1)

    print('Train loss:',running_loss)

    print('Test loss:',test_loss)

    show_img(orig, gray, conv_gray)

    train_loss.append(running_loss / running_iter)

    valid_loss.append(test_loss / test_iter)

    test_loss = 0

    running_loss = 0

    autoencoder.train()
#file_path = '..input/color-flowers/checkpoint_autoencoder.pth'

torch.save(autoencoder.state_dict(), 'checkpoint_autoencoder_graytorgb.pth')
def show_test_img(orig, gray, conv_gray):

    #orig = np.transpose(orig, (1, 2, 0))

    #gray = np.transpose(gray, (1, 2, 0))

    #gray_img = transforms.ToPILImage()

    #gray = transforms.functional.to_grayscale(gray_img(np.uint8(orig)), num_output_channels=3)

    conv_gray = np.transpose(conv_gray, (1, 2, 0))

   



    fig=plt.figure(figsize=[10,5])

    

    #orig = orig.swapaxes(0, 1).swapaxes(1, 2)

    #gray = gray.swapaxes(0, 1).swapaxes(1, 2)

    #converted_gray = conv_gray.swapaxes(0, 1).swapaxes(1, 2)

    

    # Normalize for display purpose

    #orig     = (orig - orig.min()) / (orig.max() - orig.min())

    #gray    = (gray - gray.min()) / (gray.max() - gray.min())

    #denoised = (conv_gray - conv_gray.min()) / (conv_gray.max() - conv_gray.min())

    

    fig.add_subplot(1, 3, 1, title='Original')

    plt.imshow(orig)

    

    fig.add_subplot(1, 3, 2, title='Gray')

    plt.imshow(gray,cmap='gray')

    

    fig.add_subplot(1, 3, 3, title='Converted gray')

    plt.imshow(conv_gray)

    

    fig.subplots_adjust(wspace = 0.5)

    plt.show()
import random

import numpy

img, _ = random.choice(test_data)

c_img = img





tensor = transforms.ToPILImage()

c_img = tensor(c_img)

gray = transforms.functional.to_grayscale(c_img, num_output_channels=3)



image = img.resize_((1, 3, 128, 128))





image = Variable(image).cuda()

gray_img = autoencoder(image)





show_test_img(c_img,gray,gray_img[0].data.cpu().numpy())

#show_img(c_img[0].numpy(), gray[0].data.cpu().numpy(), gray_img[0].data.cpu().numpy())