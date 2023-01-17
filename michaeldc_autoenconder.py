import matplotlib.pyplot as plt

import torch

from torch import nn

import torchvision

from torchvision import datasets

import pandas as pd

import torchvision.transforms as transforms

import torch.nn.functional as F

import numpy as  np

import seaborn as sns
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='data', train=True,

                                   download=True, transform=transform)

test_data = datasets.FashionMNIST(root='data', train=False,

                                  download=True, transform=transform)
num_workers = 0 

batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
dataiter = iter(train_loader)

images, labels = dataiter.next()

print(f"The number of images in each batch is equals: {len(images)}")
images = images.numpy()

images[0].shape
img = np.squeeze(images[0])

img.shape
fig = plt.figure(figsize = (5,5)) 

ax = fig.add_subplot(111)

ax.imshow(img, cmap='gray')

class Autoencoder(nn.Module):

    def __init__(self, dim):

        super(Autoencoder, self).__init__()

        self.encoder1 = nn.Linear(dim, 128)

        self.encoder2 = nn.Linear(128, 64)

        self.encoder3 = nn.Linear(64, 32)

        

        self.decoder1 = nn.Linear(32, 64)

        self.decoder2 = nn.Linear(64, 128)

        self.decoder3 = nn.Linear(128, dim)

        

    def forward(self, x):

        x = F.relu(self.encoder1(x))

        x = F.relu(self.encoder2(x))

        x = F.relu(self.encoder3(x))

        x = F.relu(self.decoder1(x))

        x = F.relu(self.decoder2(x))

        x = torch.sigmoid(self.decoder3(x))

        return x

    

dim = 28*28

model = Autoencoder(dim)

print(model)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


n_epochs = 2



for epoch in range(1, n_epochs+1):

    train_loss = 0.0

    

  

    for data in train_loader:

        images, _ = data

        images = images.view(images.size(0), -1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, images)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*images.size(0)

            

    train_loss = train_loss/len(train_loader)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(

        epoch, 

        train_loss

        ))
dataiter = iter(test_loader)

images, labels = dataiter.next()



images_flatten = images.view(images.size(0), -1)

output = model(images_flatten)

images = images.numpy()



output = output.view(batch_size, 1, 28, 28)

output = output.detach().numpy()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))



for images, row in zip([images, output], axes):

    for img, ax in zip(images, row):

        ax.imshow(np.squeeze(img), cmap='gray')

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)