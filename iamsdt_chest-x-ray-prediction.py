# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from PIL import Image

from matplotlib import pyplot as plt



fig = plt.figure(figsize=(55,45))

path = '../input/chest_xray/chest_xray/train/PNEUMONIA'

name = os.listdir(path)[1]

img = Image.open(path+"/"+name)

ax = fig.add_subplot(1, 5,  1, xticks=[], yticks=[])

ax.imshow(img, cmap='gray')

ax.set_title('PNEUMONIA')



path2 = '../input/chest_xray/chest_xray/train/NORMAL'

name2 = os.listdir(path2)[1]

img2 = Image.open(path2+"/"+name2)

ax = fig.add_subplot(1, 5,  2, xticks=[], yticks=[])

ax.imshow(img2, cmap='gray')

ax.set_title('NORMAL')
import random



path = '../input/chest_xray/chest_xray/train/PNEUMONIA'

li = os.listdir(path)

sizes = list()

for i in range(10):

    num = random.randint(0, len(li)-1)

    name = li[num]

    img = Image.open(path+"/"+name)

    sizes.append(img.size)



sizes
np.mean(sizes)
!wget https://raw.githubusercontent.com/Iamsdt/DLProjects/master/utils/Helper.py
!wget https://raw.githubusercontent.com/LiyuanLucasLiu/RAdam/master/cifar_imagenet/utils/radam.py
import Helper

import torch

from torchvision import datasets, transforms,models

from torch.utils.data import DataLoader



batch_size = 64

data_dir = '../input/chest_xray/chest_xray'



transform = transforms.Compose([

                                transforms.Resize(256),

                                transforms.CenterCrop(280),

                                transforms.ToTensor(),

#                                transforms.Normalize(mean, std)

])



data = datasets.ImageFolder(data_dir, transform=transform)

print(len(data))

loader = DataLoader(

    data, batch_size=batch_size)



len(loader)
mean = 0.

std = 0.

nb_samples = 0.



for images, _ in loader:

    batch_samples = images.size(0)

    data = images.view(batch_samples, images.size(1), -1)

    mean += data.mean(2).sum(0)

    std += data.std(2).sum(0)

    nb_samples += batch_samples

    break



mean /= nb_samples

std /= nb_samples



print("Mean: ", mean.numpy())

print("Std: ",std.numpy())
data_dir = '../input/chest_xray/chest_xray'



#mean = [0.485, 0.450, 0.406]

#std = [0.229, 0.224, 0.225]



train_transform = transforms.Compose([      

    transforms.Resize(256),                             

    transforms.CenterCrop(270),

    #transforms.RandomRotation(10),                             

    transforms.RandomHorizontalFlip(),                             

    transforms.ToTensor(),                             

    transforms.Normalize(mean, std)

])



test_transform = transforms.Compose([

                                transforms.Resize(256),

                                transforms.CenterCrop(280),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)

])





train_data = datasets.ImageFolder(data_dir+"/train", transform=train_transform)

test_data = datasets.ImageFolder(data_dir+"/test", transform=test_transform)



classes = train_data.classes

print(classes)





train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)



print(len(train_loader))

print(len(test_loader))
Helper.visualize(test_loader, classes)
model = models.resnet101(pretrained=True)

model.fc
# train hole network

#model = Helper.freeze_parameters(model)
import torch.nn.functional as F



def milan(input, beta=-0.25):

    '''

    Applies the Mila function element-wise:

    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))

    See additional documentation for mila class.

    '''

    return input * torch.tanh(F.softplus(input+beta))
import torch.nn as nn

from collections import OrderedDict



class mila(nn.Module):

    '''

    Applies the Mila function element-wise:

    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))

    Shape:

        - Input: (N, *) where * means, any number of additional

          dimensions

        - Output: (N, *), same shape as the input

    Examples:

        >>> m = mila(beta=1.0)

        >>> input = torch.randn(2)

        >>> output = m(input)

    '''

    def __init__(self, beta=-0.25):

        '''

        Init method.

        '''

        super().__init__()

        self.beta = beta



    def forward(self, input):

        '''

        Forward pass of the function.

        '''

        return milan(input, self.beta)
classifier = nn.Sequential(

    nn.Linear(in_features=2048, out_features=1536),

    mila(),

    nn.Dropout(p=0.4),

    nn.Linear(in_features=1536, out_features=1024),

    mila(),

    nn.Dropout(p=0.5),

    nn.Linear(in_features=1024, out_features=2),

    nn.LogSoftmax(dim=1)

)

    

model.fc = classifier

model.fc
import torch.nn.functional as F

import torchvision.transforms.functional as TF

import torch.optim as optim

import torch

import radam



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = radam.RAdam(model.parameters(), lr=0.0001)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)
optimizer
epoch = 10
model, train_loss, test_loss = Helper.train(model, train_loader, test_loader, epoch, optimizer, criterion, None)
Helper.check_overfitted(train_loss, test_loss)
Helper.test_per_class(model, test_loader, criterion, classes)
Helper.test(model, test_loader, criterion)
from PIL import Image



transform = transforms.Compose([

                                transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



def test(file):

    ids = train_loader.dataset.class_to_idx

    file = Image.open(file).convert('RGB') 

    img = transform(file).unsqueeze(0)

    with torch.no_grad():

        out = model(img.to(device))

        ps = torch.exp(out)

        top_p, top_class = ps.topk(1, dim=1)

        value = top_class.item()

        print("Value:", value)

        print(classes[value])

        plt.imshow(np.array(file))

        plt.show()
from matplotlib import pyplot as plt

path = data_dir+"/val/PNEUMONIA"

path = path+"/"+os.listdir(path)[0]

path

test(path)
path = data_dir+"/val/NORMAL"

path = path+"/"+os.listdir(path)[5]

path

test(path)
valid_transform = transforms.Compose([

                                #transforms.Resize(255),

                                #transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



valid_data = datasets.ImageFolder(data_dir+"/val", transform=valid_transform)



validloader = torch.utils.data.DataLoader(valid_data, batch_size=10)

len(validloader)
Helper.test_per_class(model, test_loader, criterion, classes)
Helper.test(model, test_loader, criterion)