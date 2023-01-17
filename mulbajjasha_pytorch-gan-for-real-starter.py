# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import torch

import torch.nn as nn

import pandas as pd

import numpy as np

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torch import autograd

from torch.autograd import Variable

from torchvision.utils import make_grid

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
train
train.iloc[:,1:]
train.iloc[:,1:].values
train.iloc[:,1:].values.astype('uint8')
images = train.iloc[:,1:].values.astype('uint8').reshape(-1,28,28)

labels = train.label.values
from PIL import Image
img = Image.fromarray(images[1])
img
import torch
class FashionMNIST(torch.utils.data.Dataset):

    def __init__(self,transform = None):

        self.images = images

        self.labels = labels

        self.transform = transform

    def __len__(self):

        return len(self.images)

    def __getitem__(self,idx):

        label = self.labels[idx]

        img = Image.fromarray(self.images[idx])

        if self.transform:

            img = self.transform(img)

        return img,label
dataset = FashionMNIST()

dataset[1][0]
import torch

from torchvision import transforms
transform = transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize(mean = (0.5,),

                                                 std = (0.5,)),

                           ])

dataset = FashionMNIST(transform = transform)
data_loader = torch.utils.data.DataLoader(dataset,batch_size = 64,shuffle = True)
import torch.nn as nn
class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.label_emb = nn.Embedding(10,10)

        self.model = nn.Sequential(

            nn.Linear(110,256),

            nn.LeakyReLU(0.2,inplace = True),

            nn.Linear(256,512),

            nn.LeakyReLU(0.2),

            nn.Linear(512,1024),

            nn.LeakyReLU(0.2),

            nn.Linear(1024,784),

            nn.Tanh(),

        )

    def __forward__(self,z,labels):

        z = z.view(z.size(0),100)

        c = self.label_emb(labels)

        x = torch.cat([z,c],1)

        out = self.model(x)

        return out.squeeze()
class Discriminator(nn.Module):

        def __init__(self):

            super().__init__()

            self.model_emb = nn.Embedding(10,10)

            self.model = nn.Sequential(

            nn.Linear(794, 1024),

            nn.LeakyReLU(0.2,inplace = True),

            nn.Dropout(0.3),

            nn.Linear(1024, 512),

            nn.LeakyReLU(0.2,inplace = True),

            nn.Dropout(0.3),

            nn.Linear(512,256),

            nn.LeakyReLU(0.2,inplace = True),

            nn.Dropout(0.3),

            nn.Linear(256,1),

            nn.Sigmoid()

            )

        def __forward__(self,z,labels):

            z = z.view(z.size(0),784)

            c = self.label_emb(labels)

            x = torch.cat([z,c],1)

            out = self.model(x)

            return out.squeeze()
generator = Generator().cuda()

discriminator = Discriminator().cuda()
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(),lr = 1e-4)

d_optimizer = torch.optim.Adam(discriminator.parameters(),lr = 1e-4)
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):

    g_optimizer.zero_grad()

    z = Variable(torch.randn(batch_size, 100)).cuda()

    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()## or you can use torch.randint(10, size = (batch_size,))

    fake_images = generator(z, fake_labels)

    validity = discriminator(fake_images, fake_labels)

    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())

    g_loss.backward()

    g_optimizer.step()

    return g_loss.item()
def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):

    d_optimizer.zero_grad()



    # train with real images

    real_validity = discriminator(real_images, labels)

    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())

    

    # train with fake images

    z = Variable(torch.randn(batch_size, 100)).cuda()

    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()

    fake_images = generator(z, fake_labels)

    fake_validity = discriminator(fake_images, fake_labels)

    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())

    

    d_loss = real_loss + fake_loss

    d_loss.backward()

    d_optimizer.step()

    return d_loss.item()
num_epoch = 30

n_critic = 5

display_step = 300

for epoch in range(num_epoch):

    print('Starting epoch {}...'.format(epoch))

    for i, (images, labels) in enumerate(data_loader):

        real_images = Variable(images).cuda()

        labels = Variable(labels).cuda()

        generator.train()

        batch_size = real_images.size(0)

        d_loss = discriminator_train_step(len(real_images), discriminator,

                                          generator, d_optimizer, criterion,

                                          real_images, labels)

        



        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)



    generator.eval()

    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))

    
z = Variable(torch.randn(100,100)).cuda()

labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)])).cuda()

sample_images = generator(z,labels).unsqueeze(1).data.cpu()

grid = make_grid(sample_images, nrow = 10, normalize = True).permute(1,2,0).numpy()

fig , ax = plt.subplot(figsize =(15,15))

ax.imshow(grid)

_ = plt.yticks([])

_ = plt.xticks(np.arrange(15,300,30),['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], rotation=45, fontsize=20)