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
# Importing the required libraries

import os

import torch

import torchvision

import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.nn.functional as F

import torchvision.models as models

from torchvision.datasets import ImageFolder

import torchvision.transforms as T

from torchvision.utils import make_grid

from torch.utils.data.dataloader import DataLoader

from torch.utils.data import random_split

%matplotlib inline
DATA_DIR = '../input/chest-xray-pneumonia/chest_xray/chest_xray'



TRAIN_DIR = DATA_DIR + '/train'                           

TEST_DIR = DATA_DIR + '/test'     

VAL_DIR = DATA_DIR + '/val'
# Let's take a look at our pictures

input_path = '../input/chest-xray-pneumonia/chest_xray/chest_xray/'



fig, ax = plt.subplots(2, 3, figsize=(15, 7))

ax = ax.ravel()

plt.tight_layout()



for i, _set in enumerate(['train', 'val', 'test']):

    set_path = input_path+_set

    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')

    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))

    ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')

    ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
# Data augmentations

image_transforms = {

    # Train uses data augmentation

    'train':

    T.Compose([

        T.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

        T.RandomRotation(degrees=15),

        T.ColorJitter(),

        T.RandomHorizontalFlip(),

        T.CenterCrop(size=224),  # Image net standards

        T.ToTensor(),

        T.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  # Imagenet standards

    ]),

    # Validation does not use augmentation

    'val':

    T.Compose([

        T.Resize(size=256),

        T.CenterCrop(size=224),

        T.ToTensor(),

        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    # Test does not use augmentation

    'test':

    T.Compose([

        T.Resize(size=256),

        T.CenterCrop(size=224),

        T.ToTensor(),

        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
# PyTorch datasets

train_ds = ImageFolder(root=TRAIN_DIR, transform=image_transforms['train'])

valid_ds = ImageFolder(root=VAL_DIR, transform=image_transforms['val'])

test_ds = ImageFolder(root=TEST_DIR, transform=image_transforms['test'])

len(train_ds), len(valid_ds), len(test_ds)
print("Classes:")

class_names = train_ds.classes

print(class_names)
batch_size = 64
# PyTorch data loaders

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
def show_image(dl):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_xticks([]); ax.set_yticks([])

        ax.imshow(make_grid(images[:1], nrow=8).permute(1, 2, 0))

        break



def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    if isinstance(target, int):

        result.append(class_names[target] + "(" + str(target) + ")")

    else:

        

        for i, x in enumerate(target):

            if (x >= threshold):

                if text_labels:

                    result.append(class_names[i] + "(" + str(i) + ")")

                else:

                    result.append(str(i))

    return ' '.join(result)



def show_sample(img, target, invert=True):

    print("target", target)

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print('Labels:', decode_target(target, text_labels=True))

        



        



show_sample(*train_ds[1545])
# Using a GPU

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
# DataLoaders

train_dl = DeviceDataLoader(train_dl, device)

valid_dl = DeviceDataLoader(valid_dl, device)
# Model

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

        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

resnet18 = models.resnet18()

resnet18
class XrayResnet(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        # Use a pretrained model

        self.network = models.resnet18(pretrained=True)

        # Replace last layer

        num_ftrs = self.network.fc.in_features

        self.network.fc = nn.Linear(num_ftrs, 10)

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))

    

    def freeze(self):

        # To freeze the residual layers

        for param in self.network.parameters():

            param.require_grad = False

        for param in self.network.fc.parameters():

            param.require_grad = True

    

    def unfreeze(self):

        # Unfreeze all layers

        for param in self.network.parameters():

            param.require_grad = True
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']



def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 

                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):

    torch.cuda.empty_cache()

    history = []

    

    # Set up cutom optimizer with weight decay

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 

                                                steps_per_epoch=len(train_loader))

    

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        lrs = []

        for batch in train_loader:

            loss = model.training_step(batch)

            train_losses.append(loss)

            loss.backward()

            

            # Gradient clipping

            if grad_clip: 

                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            

            optimizer.step()

            optimizer.zero_grad()

            

            # Record & update learning rate

            lrs.append(get_lr(optimizer))

            sched.step()

        

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        result['lrs'] = lrs

        model.epoch_end(epoch, result)

        history.append(result)

    return history
model = to_device(XrayResnet(), device)
history = [evaluate(model, valid_dl)]

history
model.freeze()
# Set hyperparameters for the model

epochs = 10

max_lr = 0.01

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
model.unfreeze()
%%time

history += fit_one_cycle(epochs, 0.001, model, train_dl, valid_dl, 

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
def plot_scores(history):

    scores = [x['val_acc'] for x in history]

    plt.plot(scores, '-x')

    plt.xlabel('epoch')

    plt.ylabel('score')

    plt.title('F1 score vs. No. of epochs');
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
def plot_lrs(history):

    lrs = np.concatenate([x.get('lrs', []) for x in history])

    plt.plot(lrs)

    plt.xlabel('Batch no.')

    plt.ylabel('Learning rate')

    plt.title('Learning Rate vs. Batch no.');
plot_lrs(history)
# Let's create a forecasting function

def predict_x_ray(image):

    xb = image.unsqueeze(0)

    xb = to_device(xb, device)

    preds = model(xb)

    prediction = preds[0]

    print("Prediction: ", prediction)

    show_sample(image, prediction)
img, target = test_ds[0]

img.shape
# Let's make predictions

predict_x_ray(test_ds[147][0])