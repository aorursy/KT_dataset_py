# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cell_images/cell_images/"))



# Any results you write to the current directory are saved as output.
import torch

from torchvision import transforms, datasets

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader

import torchvision

import matplotlib.pyplot as plt

import numpy as np

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Assuming that we are on a CUDA machine, this should print a CUDA device:



print(device)
#define transfrom to the data

data_transform = transforms.Compose(

       [transforms.Resize((108,108)),

        transforms.Pad(2),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ])



#import photo data 

cell_dataset = datasets.ImageFolder(root='../input/cell_images/cell_images/',transform=data_transform)



#define dataloader

dataset_loader = DataLoader(cell_dataset,batch_size=4, shuffle=True,num_workers=4)



split1 = int(0.1 * len(cell_dataset))

split2 = int(0.9 * len(cell_dataset))

index_list = list(range(len(cell_dataset)))

np.random.shuffle(index_list) 

test_idx = index_list[:split1]+index_list[split2:]

train_idx=index_list[split1:split2]



## create training and validation sampler objects

tr_sampler = SubsetRandomSampler(train_idx)

val_sampler = SubsetRandomSampler(test_idx)

#trainset=cell_dataset[split1:split2]



## create iterator objects for train and valid datasets

trainloader = DataLoader(cell_dataset, batch_size=4,sampler=tr_sampler,num_workers=1)

validloader = DataLoader(cell_dataset, batch_size=4,sampler=val_sampler,num_workers=1)
# functions to show an image

classes=("Parasitized","Uninfected")



def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()





# get some random training images

dataiter = iter(dataset_loader)

images, labels = dataiter.next()

#encode = autoencoder2(images)



# show images

imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
class Net3(nn.Module):

    def __init__(self):

        super(Net3, self).__init__()

        

        self.conv_layer1=nn.Sequential(

        nn.Conv2d(3, 16, 3,padding=1,bias=False),

        nn.BatchNorm2d(16,momentum=0.9),

        nn.PReLU(),

        nn.Conv2d(16, 16, 3,padding=1,bias=False),

        nn.BatchNorm2d(16,momentum=0.9),

        nn.PReLU(),

        nn.MaxPool2d(2, 2)

        )

        #x.size=16*56*56

        

        self.conv_layer2 = nn.Sequential(

        nn.Conv2d(16, 32, 3,padding=1,bias=False),

        nn.BatchNorm2d(32,momentum=0.9),

        nn.PReLU(),

        nn.Conv2d(32, 32, 3,padding=1,bias=False),

        nn.PReLU(),

        nn.Conv2d(32, 32, 3,padding=1,bias=False),

        nn.BatchNorm2d(32,momentum=0.9),

        nn.PReLU(),

        nn.MaxPool2d(2, 2)

        )

        #x.size=32*28*28

        

        self.conv_layer3 = nn.Sequential(

        nn.Conv2d(32, 64, 3,padding=1,bias=False),

        nn.BatchNorm2d(64,momentum=0.9),

        nn.PReLU(),

        nn.Conv2d(64, 64, 3,padding=1),

        nn.PReLU(),

        nn.Conv2d(64, 64, 3,padding=1,bias=False),

        nn.BatchNorm2d(64,momentum=0.9),

        nn.PReLU(),

        nn.MaxPool2d(2, 2)

        )

        #x.size=64*14*14

        

        self.conv_layer4 = nn.Sequential(

        nn.Conv2d(64, 128, 3,padding=1,bias=False),

        nn.BatchNorm2d(128,momentum=0.9),

        nn.PReLU(),

        nn.Conv2d(128, 128, 3,padding=1,bias=False),

        nn.BatchNorm2d(128,momentum=0.9),

        nn.PReLU(),

        nn.MaxPool2d(2, 2)

        )

        #x.size=128*7*7

        

        self.conv_layer5 = nn.Sequential(

        nn.Conv2d(128, 256, 3,padding=1,bias=False),

        nn.BatchNorm2d(256,momentum=0.9),

        nn.PReLU(),

        nn.Conv2d(256, 256, 3,padding=1,bias=False),

        nn.BatchNorm2d(256,momentum=0.9),

        nn.PReLU(),

        nn.MaxPool2d(2, 2)

        )

        #x.size=256*3*3

        

        self.fc_layer = nn.Sequential(

        nn.Linear(2304,1024),

        nn.PReLU(),

        nn.Linear(1024,256),

        nn.PReLU(),

        nn.Linear(256,2),

        nn.Tanhshrink()

        )

        

    def forward(self, x):

        x = self.conv_layer1(x)

        x = self.conv_layer2(x)    

        x = self.conv_layer3(x)

        x = self.conv_layer4(x)

        x = self.conv_layer5(x)

        #print(x.size())

        x = x.view(-1, 256*3*3)

        x = self.fc_layer(x)

        

        return x
def weights_init_kaiming(m):

    classname = m.__class__.__name__

    # print(classname)

    if classname.find('Conv') != -1:

        init.kaiming_normal_(m.weight, a=0, mode='fan_in')

    elif classname.find('Linear') != -1:

        init.kaiming_normal_(m.weight, a=0, mode='fan_out')

        init.constant_(m.bias, 0.0)

    elif classname.find('BatchNorm') != -1:

        init.normal_(m.weight.data, 0, 0.01)

        init.constant_(m.bias.data, 0.0)
from torch.nn import init
net3_2=Net3()

net3_2.apply(weights_init_kaiming)

#send net to GPU and train on it

net3_2.to(device)
criterion = nn.CrossEntropyLoss()

optimizer3_1 = optim.Adam(net3_2.parameters(),lr=0.0001,betas=(0.9,0.999),eps=1e-20,weight_decay=0.0002,amsgrad=True)

scheduler = ReduceLROnPlateau(optimizer3_1,mode= 'min',factor=0.1,patience=1)
for epoch in range(15):  # loop over the dataset multiple times

    

    running_loss=0

    

    for i, data in enumerate(trainloader, 0):

        # get the inputs

        inputs, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)

        

        # zero the parameter gradients

        optimizer3_1.zero_grad()



        # forward + backward + optimize

        outputs = net3_2(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer3_1.step()

        

        # print statistics

        running_loss += loss.item()

        if i % 1000 == 999:    # print every 1000 mini-batches

            #image_num=(i+1)*50

            print('[%d, %5d] loss: %.3f' %(epoch + 1, i+1, running_loss/1000))

            running_loss = 0

            

    scheduler.step(running_loss)

                  

print('Finished Training')
net3_2.cpu()
'''

for param in net3_2.parameters():

    print(param.size())

    print('{}:grad->{}'.format(param, param.grad))

'''
correct = 0

total = 0

with torch.no_grad():

    for data in trainloader:

        images, labels = data

        outputs = net3_2(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the train images: %.2f %%' % (

    100 * correct / total))
correct/total
correct = 0

total = 0

with torch.no_grad():

    for data in validloader:

        images, labels = data

        outputs = net3_2(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the test images: %.2f %%' % (

    100 * correct / total))
correct/total
class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))

with torch.no_grad():

    for data in validloader:

        images, labels = data

        outputs = net3_2(images)

        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(2):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(2):

    print("number of correct:",class_correct[i], "number of total:",class_total[i])

    print('Accuracy of %5s : %.2f %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))
torch.save(net3_2.state_dict(), 'net3_2params.pkl')
from IPython.display import FileLink, FileLinks

FileLinks('.')
2*1309/2695