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

from torchvision.utils import make_grid

import numpy as np

import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import random_split

from torchvision.transforms import ToTensor

import torch.nn as nn

import torch.nn.functional as F

%matplotlib inline
dataset = FashionMNIST(root='data/', download=True, transform = ToTensor())

test = FashionMNIST(root='data/', train=False, transform = ToTensor())
print(len(dataset))

val_size = 10000

train_size = 50000

train_ds, valid_ds = random_split(dataset, [train_size, val_size])

print(len(train_ds), len(valid_ds))
batch_size = 128

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

valid_dl = DataLoader(valid_ds, batch_size*2, shuffle=False, num_workers=4, pin_memory=True)

test_dl = DataLoader(test, batch_size*2, num_workers=4, pin_memory=True)

for images,_ in train_dl:

    print("image_size: ", images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=16).permute(1,2,0))

    break

    
def accuracy(output, labels):

    _, preds = torch.max(output, dim=1)

    return torch.tensor(torch.sum(preds==labels).item()/ len(preds))

    
class MNISTModel(nn.Module):

    def __init__(self, in_size, out_size):

        super().__init__()

        ## Hidden Layer

        self.linear1 = nn.Linear(in_size, 16)

        self.linear2 = nn.Linear(16, 32)

        self.linear3 = nn.Linear(32, out_size)

        

    def forward(self, xb):

        out = xb.view(xb.size(0), -1)

        ## First layer

        out = self.linear1(out)

        out = F.relu(out)

        ## Second Layer

        out = self.linear2(out)

        out = F.relu(out)

        ## Third Layer

        out = self.linear3(out)

        out = F.relu(out)

        return out

    

    def training_step(self, batch):

        image, label = batch

        out = self(image)

        loss = F.cross_entropy(out, label)

        return loss

    

    def validation_step(self, batch):

        image, label = batch

        out = self(image)

        loss = F.cross_entropy(out, label)

        acc = accuracy(out, label)

        return {'val_loss': loss, 'val_acc': acc}

    

    def validation_epoch_end(self, outputs):

        losses = [loss['val_loss'] for loss in outputs]

        epoch_loss = torch.stack(losses).mean()

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
torch.cuda.is_available()
def find_device():

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

device = find_device()

device
def to_device(data, device):

    if isinstance(data, (tuple, list)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)
class DeviceLoader():

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        for b in self.dl:

            yield to_device(b, self.device)

    

    def __len__(self):

        return len(self.dl)
train_loader = DeviceLoader(train_dl, device)

valid_loader = DeviceLoader(valid_dl, device)

test_loader = DeviceLoader(test_dl, device)
def evaluate(model, val_loader):

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Training Phase 

        for batch in train_loader:

            loss = model.training_step(batch)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        model.epoch_end(epoch, result)

        history.append(result)

    return history
input_size = 784

num_classes = 10
model = MNISTModel(input_size, out_size=num_classes)

to_device(model, device)
history = [evaluate(model, valid_loader)]

history
history += fit(5, 0.5, model, train_loader, valid_loader)

losses = [x['val_loss'] for x in history]

plt.plot(losses, '-x')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.title('Loss vs. No. of epochs');
def predict_image(img, model):

    xb = to_device(img.unsqueeze(0), device)

    yb = model(xb)

    _, preds  = torch.max(yb, dim=1)

    return preds[0].item()
img, label = test[0]

plt.imshow(img[0], cmap='gray')

print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])
evaluate(model, test_loader)

saved_weights_fname='fashion-feedforward.pth'

torch.save(model.state_dict(), saved_weights_fname)
