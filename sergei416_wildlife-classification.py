import torch

import torchvision

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torch.nn as nn

from torch.utils.data import random_split

import torch.nn.functional as F

from torchvision.utils import make_grid



import matplotlib.pyplot as plt

%matplotlib inline



torch.manual_seed(42)
project_name='wildlife'
dataset = ImageFolder(root='/kaggle/input/african-wildlife/', transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
dataset_size = len(dataset)

dataset_size
classes = dataset.classes

classes
num_classes = len(dataset.classes)

num_classes
test_size = 100

nontest_size = len(dataset) - test_size



nontest_ds, test_ds = random_split(dataset, [nontest_size, test_size])

len(nontest_ds), len(test_ds)
val_size = 100

train_size = len(nontest_ds) - val_size



train_ds, val_ds = random_split(nontest_ds, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 128



train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)
for images, _ in train_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))

    break
input_size = 3 * 256*256

num_classes = 4
class WildlifeModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

        

    def forward(self, xb):

        xb = xb.reshape(-1 , 3 * 256 * 256)

        out = self.linear(xb)

        return out

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss, 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    

model = WildlifeModel()
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
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
def plot_losses(history):

    losses = [x['val_loss'] for x in history]

    plt.plot(losses, '-x')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Loss vs. No. of epochs');
def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');
result0 = evaluate(model, val_loader)

result0
history1 = fit(50, 0.000005, model, train_loader, val_loader)
history2 = fit(10, 0.000001, model, train_loader, val_loader)
history3 = fit(25, 0.000001, model, train_loader, val_loader)
history4 = fit(10, 0.0000005, model, train_loader, val_loader)
history5 = fit(10, 0.0000005, model, train_loader, val_loader)
history6 = fit(10, 0.0000005, model, train_loader, val_loader)
# Replace these values with your results

history = [result0] + history1 + history2 + history3 + history4 + history5 + history6

accuracies = [result['val_acc'] for result in history]

plt.plot(accuracies, '-x')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Accuracy vs. No. of epochs');
!pip install jovian
import jovian
jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])
jovian.commit(project=project_name, environment=None)
input_size = 3 * 256 * 256

hidden_size1 = 128 # you can change this

hidden_size2 = 32

hidden_size3 = 64

hidden_size4 = 32

output_size = 4
class FeedforwardModel(nn.Module):

    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self):

        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size1)

        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        self.linear3 = nn.Linear(hidden_size2, hidden_size3)

        self.linear4 = nn.Linear(hidden_size3, hidden_size4)

        self.linear5 = nn.Linear(hidden_size4, output_size)

        

    def forward(self, xb):

        # Flatten images into vectors

        out = xb.view(xb.size(0), -1)

        # Apply layers & activation functions

        out = self.linear1(out)

        out = F.relu(out)

        out = self.linear2(out)

        out = F.relu(out)

        out = self.linear3(out)

        out = F.relu(out)

        out = self.linear4(out)

        out = F.relu(out)

        out = self.linear5(out)

        return out

    

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss, 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
model = FeedforwardModel()
for t in model.parameters():

    print(t.shape)
torch.cuda.is_available()
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')
device = get_default_device()

device
def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)
for images, labels in train_loader:

    print(images.shape)

    images = to_device(images, device)

    print(images.device)

    break
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
train_loader = DeviceDataLoader(train_loader, device)

val_loader = DeviceDataLoader(val_loader, device)
for xb, yb in val_loader:

    print('xb.device:', xb.device)

    print('yb:', yb)

    break
# Model (on GPU)

model = FeedforwardModel()

to_device(model, device)
history = [evaluate(model, val_loader)]

history
history += fit(30, 0.01000, model, train_loader, val_loader)
history += fit(30, 0.00500, model, train_loader, val_loader)
history += fit(30, 0.00010, model, train_loader, val_loader)
history += fit(30, 0.00005, model, train_loader, val_loader)
plot_losses(history)
plot_accuracies(history)
jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])
jovian.commit(project=project_name, environment=None)
import torch

import torchvision

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torch.nn as nn

from torch.utils.data import random_split

import torch.nn.functional as F

from torchvision.utils import make_grid



import matplotlib.pyplot as plt

%matplotlib inline



torch.manual_seed(42)
dataset = ImageFolder(root='/kaggle/input/african-wildlife', transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
dataset_size = len(dataset)

dataset_size
classes = dataset.classes

classes
num_classes = len(dataset.classes)

num_classes
test_size = 100

nontest_size = len(dataset) - test_size



nontest_ds, test_ds = random_split(dataset, [nontest_size, test_size])

len(nontest_ds), len(test_ds)
val_size = 100

train_size = len(nontest_ds) - val_size



train_ds, val_ds = random_split(nontest_ds, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 16



train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class CnnModel(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16



            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8



            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4



            nn.Flatten(), 

            nn.Linear(256*16*16 * 4, 1024),

            nn.ReLU(),

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Linear(512, 10))

        

    def forward(self, xb):

        return self.network(xb)
model = CnnModel()

model
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
train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

to_device(model, device);
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

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
model = to_device(CnnModel(), device)



for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    break
num_epochs1 = 10

num_epochs2 = 10

num_epochs3 = 10

opt_func = torch.optim.Adam

lr1 = 0.000010

lr2 = 0.0000005

lr3 = 0.0000001





evaluate(model, val_dl)
history = fit(num_epochs1, lr1, model, train_dl, val_dl, opt_func)
history = fit(num_epochs2, lr2, model, train_dl, val_dl, opt_func)
history = fit(num_epochs3, lr3, model, train_dl, val_dl, opt_func)
def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');
plot_accuracies(history)
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
jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])
jovian.commit(project=project_name, environment=None)
import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

from tqdm.notebook import tqdm

import torchvision.transforms as T

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

from torchvision.datasets import ImageFolder

import PIL



import matplotlib.pyplot as plt

%matplotlib inline



np.random.seed(42)
dataset = ImageFolder(root='/kaggle/input/african-wildlife/')



dataset_size = len(dataset)

dataset_size
classes = dataset.classes

classes
num_classes = len(dataset.classes)

num_classes
test_size = 100

nontest_size = len(dataset) - test_size



nontest_df, test_df = random_split(dataset, [nontest_size, test_size])

len(nontest_df), len(test_df)
val_size = 100

train_size = len(nontest_df) - val_size



train_df, val_df = random_split(nontest_df, [train_size, val_size])

len(train_df), len(val_df)
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



train_tfms = T.Compose([

    #T.RandomCrop(256, padding=8, padding_mode='reflect'),

     #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

    #T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

    T.Resize((256, 256)),

    T.RandomHorizontalFlip(), 

    T.RandomRotation(10),

    T.ToTensor(), 

     T.Normalize(*imagenet_stats,inplace=True), 

    #T.RandomErasing(inplace=True)

])



valid_tfms = T.Compose([

     T.Resize((256, 256)), 

    T.ToTensor(), 

     T.Normalize(*imagenet_stats)

])
test_df.dataset.transform = valid_tfms

val_df.dataset.transform = valid_tfms



train_df.dataset.transform = train_tfms
batch_size = 16



train_dl = DataLoader(train_df, batch_size, shuffle=True, 

                      num_workers=3, pin_memory=True)

val_dl = DataLoader(val_df, batch_size*2, 

                    num_workers=2, pin_memory=True)

test_dl = DataLoader(test_df, batch_size*2, 

                    num_workers=2, pin_memory=True)
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class CnnModel2(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        # Use a pretrained model

        self.network = models.wide_resnet101_2(pretrained=True)

        # Replace last layer

        num_ftrs = self.network.fc.in_features

        self.network.fc = nn.Linear(num_ftrs, 4)

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))





# In[40]:





model = CnnModel2()

model

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





# In[42]:





device = get_default_device()

device
train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

test_dl = DeviceDataLoader(test_dl, device)

to_device(model, device);
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

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
model = to_device(CnnModel2(), device)



for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    break
num_epochs1 = 10

opt_func = torch.optim.Adam

lr1 = 0.000010



evaluate(model, val_dl)
history = fit(num_epochs1, lr1, model, train_dl, val_dl, opt_func)
plot_accuracies(history)
plot_losses(history)
jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])
jovian.commit(project=project_name, environment=None)
model2 = CnnModel2()

model2.load_state_dict(torch.load('/kaggle/input/weights100/weights1.pth'))
model2 = to_device(model2, device)
evaluate(model2, val_dl)
evaluate(model2, test_dl)
jovian.log_metrics(val_acc=history[-1]['val_acc'], val_loss=history[-1]['val_loss'])
jovian.commit(project=project_name, environment=None)