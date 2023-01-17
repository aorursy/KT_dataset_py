%matplotlib inline
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

from torchvision.utils import make_grid
from torch.utils.data import random_split
import os

import matplotlib.pyplot as plt
DATA_DIR = '../input/dogs-cats-images/dataset'
TRAIN_DIR = DATA_DIR + "/training_set"
TEST_DIR = DATA_DIR + "/test_set"

classes = os.listdir(TRAIN_DIR)
print(classes)
cats = os.listdir(TRAIN_DIR + "/cats")
print('No. of training examples for cats:', len(cats))
print(cats[:5])
dogs = os.listdir(TRAIN_DIR + "/dogs")
print('No. of training examples for dogs:', len(dogs))
print(dogs[:5])
cats = os.listdir(TEST_DIR + "/cats")
print('No. of testing examples for cats:', len(cats))
print(cats[:5])
cats = os.listdir(TEST_DIR + "/dogs")
print('No. of testing examples for dogs:', len(dogs))
print(dogs[:5])
from torchvision.datasets import ImageFolder
transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
dataset = ImageFolder(DATA_DIR+'/training_set', transform=transform)
test_dataset = ImageFolder(DATA_DIR+'/test_set', transform=transform)

img, label = dataset[0]
print(img.shape, label)
img
print(dataset.classes)
import matplotlib.pyplot as plt

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
show_example(*dataset[0])
show_example(*dataset[4800])
random_seed = 42
torch.manual_seed(random_seed);
len(dataset)
val_size = 2000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size=64
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)
def show_batch(dl, invert=False):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break
show_batch(train_loader)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class ImageClassificationModelBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.cross_entropy(out, targets)  # Calculate loss
        acc = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,result['train_loss'], result['val_loss'], result['val_acc']))
class ImageClassificationModel(ImageClassificationModelBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #output 32 X 100 X 100 | (Receptive Field (RF) -  3 X 3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   #output 64 X 100 X 100 | RF 5 X 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 50 x 50 | RF 10 X 10

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 64 x 50 x 50 | RF 12 X 12
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 25 x 25  | RF 24 X 24
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 x 25 x 25  | RF 26 X 26
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 12 x 12 | RF 52 X 52
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1), #512* 10* 10 | RF 54 X 54
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 512 x 5 x 5 | RF - 108X 108
            
#             nn.Conv2d(512, 1024, kernel_size=3, stride=1),#1024 X 3 X 3 | Rf - 110 X 110
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 1024 x 4 x 4
            
#             nn.Conv2d(1024, 2048, kernel_size=3, stride=1), # output: 2048 x 2 x 2


            nn.Flatten(),
            nn.Linear(512 * 5 * 5,10))
#             nn.Linear(1024 * 3 * 3, 10))
         
    def forward(self, xb):
        return self.network(xb)
#function to ensure that our code uses the GPU if available, and defaults to using the CPU if it isn't.
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
# a function that can move data and model to a chosen device.    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


#Finally, we define a DeviceDataLoader class to wrap our existing data loaders and move data to the selected device, 
#as a batches are accessed. Interestingly, we don't need to extend an existing class to create a PyTorch dataloader. 
#All we need is an __iter__ method to retrieve batches of data, and an __len__ method to get the number of batches.

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
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
model = to_device(ImageClassificationModel(), device)


!pip install torchsummary
from torchsummary import summary
# print the summary of the model
summary(model, input_size=(3, 100, 100), batch_size=-1)
history = [evaluate(model, val_loader)]
history
num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.001
history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)
def plot_scores(history):
#     scores = [x['val_score'] for x in history]
    acc = [x['val_acc'] for x in history]
    plt.plot(acc, '-x')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('acc vs. No. of epochs');
plot_scores(history)

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
plot_losses(history)

