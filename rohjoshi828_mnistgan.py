# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

#Importing the dataset from kaggle directory 
dataset = pd.read_csv("../input/digit-recognizer/train.csv")

#targets
data = dataset.drop("label", axis = 1)

#features in pixel values
labels = dataset['label']
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

#transfering the dataset to a tensor dataset
tensor_dataset = TensorDataset(torch.Tensor(np.array(data)), torch.from_numpy(np.array(labels)))

#creating a data loader with batch size 64 and shuffling the dataset
data_loader = DataLoader(tensor_dataset, batch_size = 1, shuffle = True)

import torch.nn as nn
import torch.nn.functional as F

#creating the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        pred = F.relu(self.fc1(x))
        pred = F.relu(self.fc2(pred))
        pred = F.relu(self.fc3(pred))
        pred = torch.sigmoid(self.fc4(pred))
        return pred
    
#creating the generator network
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 384)
        self.fc3 = nn.Linear(384, 512)
        self.fc4 = nn.Linear(512, 784)
        
    def forward(self, x):
        pred = F.relu(self.fc1(x))
        pred = F.relu(self.fc2(pred))
        pred = F.relu(self.fc3(pred))
        pred = F.tanh(self.fc4(pred))
        return pred
        
#creating the noise for the generator network
def noise():
    arr = torch.randn(100)
    return arr

#creates targets for discriminator
def real_targets(size):
    return torch.ones(size)

def fake_targets(size):
    return torch.zeros(size)
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

#create generator and discriminator classes
discriminator = Discriminator()
generator = Generator()

#create optimizers for both neworks
d_optim = optim.Adam(discriminator.parameters(), lr = 0.0002)
g_optim = optim.Adam(generator.parameters(), lr = 0.0002)

#loss function is binary cross entropy
loss_function = nn.BCELoss()

generator_losses = []
discriminator_losses = []

num_iterations = 0

for data, label in data_loader:
    
    ### Training The Discriminator ###
    
    #the discriminator output on the true images
    real_output = discriminator(data)
    
    #the discriminator loss on predicting the true images
    d_real_loss = loss_function(real_output, real_targets(1))
    
    #back propogating the discriminator loss on true images
    d_real_loss.backward()

    #the discriminator output on fake images
    fake_output = discriminator(generator(noise()))
    
    #the discriminator loss on predicting fake images
    d_fake_loss = loss_function(fake_output, fake_targets(1))
    
    #back propogating the disciminator loss on fake images
    d_fake_loss.backward()
    
    d_optim.step()
    
    ### Training the Generator ###
    
    #the discriminator output on fake images
    fake_output = discriminator(generator(noise()))
    
    #the generator loss is how good the discriminator was able to predict the images
    generator_loss = loss_function(fake_output, real_targets(1))
    
    #back propgation on generator loss
    generator_loss.backward()
    g_optim.step()
    
    g_optim.zero_grad()
    d_optim.zero_grad()
               
    #number of iterations increases
    num_iterations+=1
    
    #prints the iteration if it is a multiple of 100
    if num_iterations%100 == 0:
        print(num_iterations)
    
    #Appends the loss to the loss lists                            
    discriminator_losses.append(d_fake_loss+d_real_loss)
    generator_losses.append(generator_loss)
    
    
    
    
    
from matplotlib import pyplot as plt
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
np.array(discriminator(data).detach().numpy())

plt.plot(range(len(generator_losses)), generator_losses)
plt.plot(range(len(generator_losses)), generator_losses)
plt.show()
nn.BCELoss(real_output, real)
real_output.reshape(-1,len(data), )
discriminator(generator(noise()))
g = Generator()
g(noise())
