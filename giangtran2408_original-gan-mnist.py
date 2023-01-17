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
df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_train
df_train.describe()
X_train = df_train.drop("label", axis=1).values

y_train = df_train["label"].values
X_train = X_train/255.0
import torch

import torch.nn as nn

import matplotlib.pyplot as plt

from torchvision.utils import make_grid

%matplotlib inline
class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()



        def block(in_feat, out_feat, normalize=True):

            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:

                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers



        self.model = nn.Sequential(

            *block(100, 128, normalize=False),

            *block(128, 256),

            *block(256, 512),

            nn.Linear(512, 784),

            nn.Tanh()

        )



    def forward(self, z):

        img = self.model(z)

        return img





class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()



        self.model = nn.Sequential(

            nn.Linear(784, 512),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),

            nn.Sigmoid(),

        )



    def forward(self, img):

        img = self.model(img)

        return img





class TrainerGAN:



    def __init__(self, generator, discriminator, batch_size, num_iterations, optimizer_G, optimizer_D, adversarial_loss):

        self.generator = generator

        self.discriminator = discriminator

        self.batch_size = batch_size

        self.num_iterations = num_iterations

        self.optimizer_G = optimizer_G

        self.optimizer_D = optimizer_D

        self.adversarial_loss = adversarial_loss

        

    def train(self, X_train, y_train):

        m = X_train.shape[0]

        d = 100

        print_loss = 100

        k = 4

        

        for i in range(self.num_iterations):

            ind = np.random.choice(m, self.batch_size)

            report_loss = 0.0

            ones = torch.ones(self.batch_size, 1)

            zeros = torch.zeros(self.batch_size, 1)

            z = torch.empty(self.batch_size, d).normal_(0, 1)

            z = self.generator(z)



            for _ in range(k):

                x = X_train[ind]

                

                Dz = self.discriminator(z.detach())

                Dx = self.discriminator(x)

                

                optimizer_D.zero_grad()

                D_loss = self.adversarial_loss(Dx, ones) + self.adversarial_loss(Dz, zeros)

                D_loss.backward()

                optimizer_D.step()

                

                report_loss += D_loss.item()

            

            report_loss = report_loss / k

            

            Dz = self.discriminator(z)

            optimizer_G.zero_grad()

            G_loss = self.adversarial_loss(Dz, ones)

            G_loss.backward()

            optimizer_G.step()

            

            if (i+1) % print_loss == 0:

                print("[Iteration: %d] [Discriminator loss: %.5f] [Generator loss: %.5f]" % (i+1, report_loss, G_loss.item()))

                z = torch.empty(self.batch_size, d).normal_(0, 1)

                with torch.no_grad():

                    z = self.generator(z)

                z = torch.reshape(z, (self.batch_size, 1, 28, 28))

                imgs = make_grid(z, normalize=True, range=(0, 1))

                imgs = imgs.numpy()

                plt.imshow(np.transpose(imgs, (1, 2, 0)), interpolation="nearest")

                plt.show()
generator = Generator()

discriminator = Discriminator()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))
loss_func = nn.BCELoss()

batch_size = 64

num_iterations = 10000

trainer = TrainerGAN(generator, discriminator, batch_size, num_iterations, optimizer_G, optimizer_D, loss_func)
trainer.train(torch.Tensor(X_train), y_train)