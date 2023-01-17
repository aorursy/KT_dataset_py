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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import torch

from torch.utils.data import DataLoader, Dataset

import torchvision

from torchvision import transforms
class ToTensor(object):

    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range

    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    """



    def __call__(self, pic):

        """

        Args:

            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:

            Tensor: Converted image.

        """

        return F.to_tensor(pic)



    def __repr__(self):

        return self.__class__.__name__ + '()'
class Dataset(object):

    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override

    ``__len__``, that provides the size of the dataset, and ``__getitem__``,

    supporting integer indexing in range from 0 to len(self) exclusive.

    """



    def __getitem__(self, index):

        raise NotImplementedError



    def __len__(self):

        raise NotImplementedError



    def __add__(self, other):

        return ConcatDataset([self, other])


class DatasetMNIST2(Dataset):

    

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        # load image as ndarray type (Height * Width * Channels)

        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]

        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)

        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))

        label = self.data.iloc[index, 0]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
traindataset = DatasetMNIST2('../input/train.csv', transform=torchvision.transforms.ToTensor())

img, lab = traindataset.__getitem__(0)



print('image shape at the first row : {}'.format(img.size()))
trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)



dataiter = iter(trainloader)

images, labels = dataiter.next()

print(type(images))



print('images shape on batch size = {}'.format(images.size()))

print('labels shape on batch size = {}'.format(labels.size()))
plt.imshow(images[1].numpy().squeeze(),cmap='OrRd_r')
x=torch.Tensor

def activation(x):

  return (1/(1+torch.exp(-x)))
feature=images.view(images.shape[0],-1)#matrix of 1*3

n_input=784

n_hidden=256 #hidden layer

n_output=10 #output layer

w1=torch.randn(n_input,n_hidden) #random weight generation from [input * hidden layer ] size matrix

w2=torch.randn(n_hidden,n_output) #random second weight generation[hidden *output layer] size matrix

b1=torch.randn((1,n_hidden)) #randoom bias generation from [single layer of column of hidden size]matrix

b2=torch.randn((1,n_output)) #random bias generation from [single layer of column of output size]matrix

print(w1,w2,b1,b2)
h=activation(torch.matmul(feature,w1)+b1) 

output=activation(torch.matmul(h,w2)+b2)

print(output)
from torch import nn
class Network(nn.Module):

  def __init__(self):

    super().__init__()

    #input to hiden layers

    self.hidden=nn.Linear(784,256)

    #input to ooutput layers

    self.output=nn.Linear(256,10)

    self.sigmoid=nn.Sigmoid()

    self.softmax=nn.Softmax(dim=1)

  def forward(self,x):

    x=self.hidden(x)

    x=self.output(x)

    x=self.sigmoid(x)

    x=self.softmax(x)

    retun(x)
import torch.nn.functional as F

from torch import optim
class Network1(nn.Module):

  def __init__(self):

    super().__init()

    #input to hidden layers

    self.hidden=nn.Linear(784,256)

    #inpput to output layers

    self.output=nn.Linear(256,10)

  def forward(self,x):

    x=F.Sigmoid(self.hidden)

    x=F.Softmax(sel.output)

    return x
model=nn.Sequential(nn.Linear(784,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,10),nn.ReLU(),nn.LogSoftmax(dim=1))

criterion=nn.NLLLoss()

optimizer=optim.SGD(model.parameters(),lr=0.5)#traning data

epochs=13

for e in range(epochs):

  running_loss=0

  for images,labels in trainloader:

    images=images.view(images.shape[0],-1)

    optimizer.zero_grad()

    output=model.forward(images)

    loss=criterion(output,labels)

    loss.backward()

    optimizer.step()

    running_loss+=loss.item()

  else:

    print({running_loss/len(trainloader)})
def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

    

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
%matplotlib inline

images,labels=next(iter(trainloader))

img=images[0].view(1,784)

with torch.no_grad():

  logits=model.forward(img)

ps=F.softmax(logits,dim=1)

view_classify(img.view(1,28,28),ps)