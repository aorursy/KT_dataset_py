!pip install fastai2
from fastai2.vision.all import *

import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torchvision.transforms as T

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

%matplotlib inline
project_name = 'Dance-Classifier'
DATA_DIR = '../input/indian-danceform-classification/dataset'



TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images

TEST_DIR = DATA_DIR + '/test'                             # Contains test images



TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images

TEST_CSV =  DATA_DIR + '/test.csv'                        # Contains dummy labels for test image
os.listdir(TRAIN_DIR)[:5]
Image.open(TRAIN_DIR+'/234.jpg')
Image.open(TRAIN_DIR+'/287.jpg')
len(os.listdir(TRAIN_DIR))
train_df = pd.read_csv(TRAIN_CSV)

train_df.head()
train_df.target.value_counts()
def get_x(r): return DATA_DIR+'/train/'+r['Image']  # Image Directory

def get_y(r): return r['target']                    # Getting the label

dblock = DataBlock(

    blocks=(ImageBlock,CategoryBlock),

    splitter=RandomSplitter(),

    get_x = get_x,

    get_y = get_y,

    item_tfms = Resize(330),

    batch_tfms=aug_transforms(mult=2))



dls = dblock.dataloaders(train_df)



train_dl = dls.train

valid_dl = dls.valid
dls.show_batch()
dls.train.show_batch()
dls.valid.show_batch()
# Let's save the work to jovian

!pip install jovian --upgrade -q

import jovian

jovian.commit(project = project_name, environment=None)
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

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
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
class DanceModel(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        self.linear1 = nn.Linear(input_size, 512)  # first linear layer

        self.linear2 = nn.Linear(512, 128)          # second linear layer

        self.linear3 = nn.Linear(128, output_size)  # third linear layer



        

    def forward(self, xb):

        # Flatten images into vectors

        out = xb.view(xb.size(0), -1)

        # Apply layers & activation functions

        out = self.linear1(out)

        out = F.relu(out)

        

        out = self.linear2(out)

        out = F.relu(out)

        

        out = self.linear3(out)

        return out
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
device = get_default_device()

print(device)



# image size is 224x224x3 

# output class = 8

input_size = 330*330*3

output_size = 8

model = to_device(DanceModel(), device)
history = [evaluate(model, valid_dl)]

history
history += fit(5, 1e-3, model, train_dl, valid_dl)
history += fit(5, 1e-2, model, train_dl, valid_dl)
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

    
plot_losses(history)
plot_accuracies(history)
arch1 = "4 layers (1024, 512, 128, 8)"

arch2 = '3 layers (512, 128, 8)'

arch = [arch1, arch2]
lrs1 = [1e-2, 1e-3]

lrs2 = [1e-2, 1e-3]

lrs = [lrs1, lrs2]
epoch1 = [5, 5]

epoch2 = [5, 5]

epochs = [epoch1, epoch2]
valid_acc = [14.8, 24]

valid_loss = [2.10, 2.10]
torch.save(model.state_dict(), 'dance-feed-forward.pth')
# Clear previously recorded hyperparams & metrics

jovian.reset()
jovian.log_hyperparams(arch=arch, 

                       lrs=lrs, 

                       epochs=epochs)
jovian.log_metrics(valid_loss=valid_loss, valid_acc=valid_acc)
jovian.commit(project=project_name, outputs=['dance-feed-forward.pth'], environment=None)
class DanceResnet(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        # Use a pretrained model

#         self.network = models.resnet34(pretrained=True)

        self.network = models.resnet50(pretrained=True)

        

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

        for batch in tqdm(train_loader):

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
device = get_default_device()

print(device)

model = to_device(DanceResnet(), device)
history = [evaluate(model, valid_dl)]

history
model.freeze()
epochs = 5

max_lr =  1e-3

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

history += fit_one_cycle(5, 1e-4, model, train_dl, valid_dl, 

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
plot_losses(history)
plot_accuracies(history)
arch1 = 'resnet 34'

arch2 = 'resnet 50'

arch3 = 'resnet 50: replaced RandomResized224 to Resize224'

arch3 = 'resnet 50: replaced Resized224 to Resize330'

arch = [arch1, arch2, arch3]
lrs1 = [1e-4, 1e-4]

lrs2 = [1e-3, 1e-4]

lrs = [lrs1, lrs2]
epoch1 = [5, 5]

epoch2 = [5, 5]

epochs = [epoch1, epoch2]
valid_acc = [64, 71]

valid_loss = [1.76, 1.68]
torch.save(model.state_dict(), 'dance-resnet50.pth')
# Clear previously recorded hyperparams & metrics

jovian.reset()
jovian.log_hyperparams(arch=arch, 

                       lrs=lrs, 

                       epochs=epochs)
jovian.log_metrics(valid_loss=valid_loss, valid_acc=valid_acc)
jovian.commit(project=project_name, outputs=['dance-resnet50.pth'], environment=None)