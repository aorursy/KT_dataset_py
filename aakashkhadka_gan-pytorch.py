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
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
batch_size=100
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
mnist=torchvision.datasets.MNIST(root='datasets/',train=True,transform=transform,download=True)
!ls
mnist_loader=torch.utils.data.DataLoader(mnist,shuffle=True,batch_size=batch_size)
img,labels=iter(mnist_loader).next()
labels
grid=torchvision.utils.make_grid(img)
grid=grid.detach().numpy()
grid=np.transpose(grid,(1,2,0))
plt.imshow(grid)
#latent size is the latent variable vector used in generator
latent_size=64
#hidden size in discriminator and generator
hidden_size=256
image_size=28*28
epochs=100
#Discriminator
D=nn.Sequential(
    nn.Linear(image_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.5),
    nn.Linear(hidden_size,hidden_size),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.5),
    nn.Linear(hidden_size,1),
    nn.Sigmoid()
)
#Generator
G=nn.Sequential(
    nn.Linear(latent_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size,image_size),
    nn.Tanh()
)
D=D.to(device)
G=G.to(device)
bce_loss=nn.BCELoss()
d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0002)
g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0002)
for epoch in range(epochs):
    for i ,(images,_) in enumerate(mnist_loader):
        img=images.reshape(batch_size,-1).to(device)
        real_labels=torch.ones(batch_size,1).to(device)
        fake_labels=torch.zeros(batch_size,1).to(device)
        
        outputs=D(img)
        
        d_loss_real=bce_loss(outputs,real_labels)
        real_score=outputs
        
        z=torch.randn(batch_size,latent_size).to(device)
        fake_images=G(z)
        
        outputs=D(fake_images)
        d_loss_fake=bce_loss(outputs,fake_labels)
        fake_score=outputs
        
        d_loss=d_loss_fake+d_loss_real
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        
        d_loss.backward()
        d_optimizer.step()
        
        #now updating generator
        z=torch.randn(batch_size,latent_size).to(device)
        fake_images=G(z)
        
        outputs=D(fake_images)
        g_loss=bce_loss(outputs,real_labels)
        
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if i % 200==0:
            print(f'Epoch {i} d_loss {d_loss.item()} g_loss {g_loss.item()}')
            
    fake_images=fake_images.reshape(fake_images.size(0),1,28,28)
img=torchvision.utils.make_grid(fake_images)
img=img.detach().cpu().numpy()
img=img.clip(0,1)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(img,(1,2,0)))

