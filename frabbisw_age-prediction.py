import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F
from matplotlib import pyplot as plt

import numpy as np
batch_size=8
transform = transforms.Compose(

    [transforms.Resize((128,128)),

     transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset=torchvision.datasets.ImageFolder("/kaggle/input/facial-age/face_age/",transform=transform)

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=2)
def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 29 * 29, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 1)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 29 * 29)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        return x
model=Model().cuda()

criterion=nn.MSELoss()

optim=torch.optim.Adam(model.parameters(),lr=.001)
dataiter = iter(dataloader)

x, y = dataiter.next()

x=x.cuda()

y=y.reshape((batch_size,1)).cuda().type(torch.float32)/100

print(y,y.dtype)
import datetime

timea = datetime.datetime.now()

for i in range(100000):

    dataiter = iter(dataloader)

    x, y = dataiter.next()

    x=x.cuda()

    y=.01+y.reshape((batch_size,1)).cuda().type(torch.float32)/100

    z=model(x)



    loss=criterion(z,y)

    optim.zero_grad()

    loss.backward()

    optim.step()



    if i % 10000 == 0:

        print(i, loss)

        

timeb = datetime.datetime.now()

print("time in seconds: ",(timeb-timea).seconds)
for i in range(10):

    dataiter = iter(dataloader)

    x, y = dataiter.next()

    x=x.cuda()

    y=.01+y.reshape((batch_size,1)).cuda().type(torch.float32)/100

    z=model(x)



    imshow(torchvision.utils.make_grid(x.cpu()))

    y=(y*100).type(torch.int64)

    z=(z*100).type(torch.int64)

    print(y.reshape(1,8).tolist()[0])

    print(z.reshape(1,8).tolist()[0])
torch.save({'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict}, 'checkpoint.pt.tar.gz')
# model=Model().cuda()

# model.load_state_dict(torch.load("model.pt"))

# model.eval()