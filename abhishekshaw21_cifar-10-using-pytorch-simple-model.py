# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn.functional as F

from torchvision import datasets,transforms

from torch import nn

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

#from tqdm.notebook import tqdm

from tqdm import tqdm
# # Dowload the dataset

# from torchvision.datasets.utils import download_url

# dataset_url = "http://files.fast.ai/data/cifar10.tgz"

# download_url(dataset_url, '.')

# import tarfile

# # Extract from archive

# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:

#     tar.extractall(path='./data')
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor
data_dir = '/kaggle/input/cifar10-pngs-in-folders/cifar10/cifar10'

dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

# print(dataset)
dataset
random_seed = 21

torch.manual_seed(random_seed);

from torch.utils.data import random_split

val_size = 5000

train_size = len(dataset) - val_size



train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
# x = [4500]*10

# for p in train_ds:

#     x[p[1]] = x[p[1]] - 1

# x
# y = [500]*10

# for p in val_ds:

#     y[p[1]] = y[p[1]] - 1

# y
from torch.utils.data.dataloader import DataLoader



batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
import torch.nn as nn

import torch.nn.functional as F
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
device = get_default_device()

device
@torch.no_grad()

def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        accu = accuracy(out,labels)

        return loss,accu

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        _,preds = torch.max(out, dim=1)

        return {'Loss': loss.detach(), 'Accuracy': acc,'Entropy': preds}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['Loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['Accuracy'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        classCount = torch.zeros(10, dtype=torch.float64, device="cuda")

        for x in outputs:

            for y in x['Entropy']:

                classCount[y] += 1

        entropy = 0

        classCount /= sum(classCount)

        for pi in classCount:

            if(pi!=0):

                entropy -= pi*torch.log(pi)

        return {'Loss': epoch_loss.item(), 'Accuracy': epoch_acc.item(),'Entropy':entropy}

    

    def epoch_end(self, epoch, result):

        print("Epoch :",epoch + 1)

        print(f'Train Accuracy:{result["train_accuracy"]*100:.2f}% Validation Accuracy:{result["Accuracy"]*100:.2f}%')

        print(f'Train Loss:{result["train_loss"]:.4f} Validation Loss:{result["Loss"]:.4f}')

        print(f'Validation Entropy:{result["Entropy"]:.2f}')
class Cifar10CnnModel(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 24, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            

            nn.Flatten(), 

            nn.Linear(32*32*32, 256),

            nn.ReLU(),

            nn.Linear(256, 10),

            nn.ReLU()

        )

    def forward(self, xb):

        return self.network(xb)
model = Cifar10CnnModel()
train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

to_device(model, device)
@torch.no_grad()

def evaluate(model, data_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in data_loader]

    return model.validation_epoch_end(outputs)



def fit(model, train_loader, val_loader,epochs=25,learning_rate=0.06):

    best_valid = None

    history = []

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,weight_decay=0.001)

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        train_accuracy = []

        for batch in tqdm(train_loader):

            loss,accu = model.training_step(batch)

            train_losses.append(loss)

            train_accuracy.append(accu)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        result['train_accuracy'] = torch.stack(train_accuracy).mean().item()

        model.epoch_end(epoch, result)

        if(best_valid == None or best_valid<result['Accuracy']):

            best_valid=result['Accuracy']

            torch.save(model.state_dict(), 'cifar10-cnn.pth')

        history.append(result)

    return history
history = fit(model, train_dl, val_dl)
def plot_accuracies(history):

    Validation_accuracies = [x['Accuracy'] for x in history]

    Training_Accuracies = [x['train_accuracy'] for x in history]

    plt.plot(Training_Accuracies, '-rx')

    plt.plot(Validation_accuracies, '-bx')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.legend(['Training', 'Validation'])

    plt.title('Accuracy vs. No. of epochs');

plot_accuracies(history)
def plot_losses(history):

    train_losses = [x.get('train_loss') for x in history]

    val_losses = [x['Loss'] for x in history]

    plt.plot(train_losses, '-bx')

    plt.plot(val_losses, '-rx')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['Training', 'Validation'])

    plt.title('Loss vs. No. of epochs');

plot_losses(history)
final_model = Cifar10CnnModel()

final_model.load_state_dict(torch.load('/kaggle/working/cifar10-cnn.pth'))

to_device(final_model, device);
test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())

test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size), device)

result = evaluate(final_model, test_loader)

print(f'Test Accuracy:{result["Accuracy"]*100:.2f}%')

print(f'Entropy:{result["Entropy"]:.2f}')