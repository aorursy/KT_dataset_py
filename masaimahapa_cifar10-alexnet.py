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
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import time
transform= transforms.Compose(
    [   transforms.Resize(256),
     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223 , 0.24348513, 0.26158784])
    ])
trainset= torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform,
                                      download=True)
testset= torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform,
                                     download=True)
trainset.data.shape
train_means= trainset.data.mean(axis=(0,1,2))/255
train_means
train_stds= trainset.data.std(axis=(0,1,2))/255
train_stds
trainloader= torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,
                                        num_workers=8)
testloader= torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False,
                                       num_workers=8)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
from torch.utils.data import Subset
def train_valid_split(dl, val_split=0.25):
    total_items= dl.dataset.data.shape[0]
    idxs= np.random.permutation(total_items)
    train_idxs, valid_idxs= idxs[round(total_items*val_split):], idxs[:round(total_items*val_split)]
    
    train= Subset(dl, train_idxs)
    valid= Subset(dl, valid_idxs)
    return train, valid
train_dl, valid_dl= train_valid_split(trainloader)
import matplotlib.pyplot as plt
def show_image(img):
    img= img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
show_image(torchvision.utils.make_grid(images[:4]))
[classes[each] for each in labels[:4]]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# trainloader.to(device);
print(device)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        #1
        self.features= nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #2
        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #3
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        #4
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        #5
        nn.Conv2d(384, 256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool= nn.AvgPool2d(6)
        self.classifier= nn.Sequential(
            nn.Dropout(), nn.Linear(256*6*6, 4096), #128*2*2, 1024
        nn.ReLU(inplace=True), nn.Dropout(),
        nn.Linear(4096, num_classes))
        
    def forward(self, x):
        x= self.features(x)
        x=x.view(x.size(0), 256*6*6)
        x= self.classifier(x)
        return x
model= AlexNet(num_classes=10).to(device)
model
#loss function and optimizer
criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(params= model.parameters(), lr=3e-4)
import datetime

def convert_seconds_format(n):
    return str(datetime.timedelta(seconds =n))

all_losses=[]
all_valid_losses=[]
print('training starting...')
start_time= time.time()
for epoch in range(10):
    epoch_start=time.time()
    model.train()
    running_loss= 0.0
    running_valid_loss=0.0
    predictions=[]
    total=0
    correct=0
    
    for i, data in enumerate(train_dl.dataset, 0):

        inputs, labels= data[0].to(device), data[1].to(device)

        #zero parameter gradients
        optimizer.zero_grad()

        #forward + back optimize
        outputs= model(inputs)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #stats
        running_loss += loss.item()
    all_losses.append(running_loss/i)
    
    #evaluation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_dl.dataset, 0):
            inputs, labels= data[0].to(device), data[1].to(device)
            outputs= model(inputs)
            valid_loss= criterion(outputs, labels)
            running_valid_loss+= valid_loss.item()
            
            #the class with the highest score
            _, predicted= torch.max(outputs.data, 1)
            predictions.append(outputs)
            total+= labels.size(0)
            correct+= (predicted==labels).sum().item()
    epoch_end=time.time()
    epoch_time= convert_seconds_format(epoch_end-epoch_start)
    
    all_valid_losses.append(valid_loss)
    print(f"epoch {epoch+1}, running loss: {all_losses[-1]}")
    print(f"validation accuracy: {correct/total}. validation loss: {all_valid_losses[-1]}")
    print(f"epoch time: {epoch_time}")
end_time= time.time()
train_time= convert_seconds_format(end_time- start_time)
print('training complete')
print(f"total time to train: {train_time}")

x_axis=[i for i in range(1, 11)]
x_axis
valid_losses_list=[each.item() for each in all_valid_losses]
    
plt.plot(x_axis, all_losses, label='train')
plt.plot(x_axis, valid_losses_list, label='valid')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend();
correct, total=0, 0
predictions=[]

model.eval();
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels= data[0].to(device), data[1].to(device)
        #inputs= inputs.view(-1, 32*32*3)
        outputs= model(inputs)
        #the class with the highest score
        _, predicted= torch.max(outputs.data, 1)
        predictions.append(outputs)
        total+= labels.size(0)
        correct+= (predicted==labels).sum().item()
print(f' Accuracy score of: {correct/total}')
