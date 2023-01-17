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

import matplotlib.pyplot as plt

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls ../input/sarscov2-ctscan-dataset
import torch

from torch import nn

import torchvision

import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
class Net(nn.Module):

  def __init__(self):

    super(Net,self).__init__()

    self.densenet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)

    for p in self.densenet.parameters():

           p.requires_grad = False

    self.densenet.classifier = nn.Identity()

    self.flatten1 = nn.Flatten()

    self.linear1 = nn.Linear(1920,128)

    self.relu1 = nn.ReLU()

    self.dropout1 = nn.Dropout(0.2)

    self.linear2 = nn.Linear(128,64)

    self.relu2 = nn.ReLU()

    self.dropout2 = nn.Dropout(0.3)

    self.linear3 = nn.Linear(64,2)

    self.sigmoid = nn.Sigmoid()

    

    

  def forward(self,x):

    x=self.densenet(x)

    x=self.flatten1(x)

    x=self.linear1(x)

    x=self.relu1(x)

    x=self.dropout1(x)

    x=self.linear2(x)

    x=self.relu2(x)

    x=self.dropout2(x)

    x=self.linear3(x)

    x=self.sigmoid(x)

    return x

model = Net().to('cpu')
model(torch.rand((1,3,224,224)))
model
model
transform = transforms.Compose(

    [

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
dataset = torchvision.datasets.ImageFolder('../input', transform=transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=20,

                                          shuffle=True, num_workers=2)

images, labels = next(iter(trainloader))
images
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
losses = []

for epoch in range(5):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0) 

        # print statistics

        # running_loss += loss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 2000))

            running_loss = 0.0

        epoch_loss = running_loss / len(trainloader)

        losses.append(epoch_loss)

        print(f"Epoch:{epoch} Loss:{loss}")

    writer.add_scalar("Loss/train", epoch_loss, epoch)

    print(f"Epoch:{epoch} Loss:{loss} Epoch Loss:{epoch_loss}")

print('Finished Training')
plt.plot(losses)
grid = torchvision.utils.make_grid(images)

writer.add_image('images', grid, 0)

writer.add_graph(model, images)

writer.close()

!tar -cvf file.tar.xz ./runs
from IPython.display import FileLink

FileLink('./file.tar.xz')
!ls
