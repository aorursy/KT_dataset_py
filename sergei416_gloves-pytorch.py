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

torch.manual_seed(42)
dataset = ImageFolder(root='/kaggle/input/medical-gloves-recognition-dataset/dataset/')



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

    T.RandomCrop(128, padding=8, padding_mode='reflect'),

     #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

    T.Resize((128, 128)),

    T.RandomHorizontalFlip(), 

     T.RandomVerticalFlip(),

    T.RandomRotation(10),

    T.ToTensor(), 

     T.Normalize(*imagenet_stats,inplace=True), 

    #T.RandomErasing(inplace=True)

])



valid_tfms = T.Compose([

     T.Resize((128, 128)), 

    T.ToTensor(), 

     T.Normalize(*imagenet_stats)

])




test_df.dataset.transform = valid_tfms

val_df.dataset.transform = valid_tfms



train_df.dataset.transform = train_tfms







batch_size = 64



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

        self.network.fc = nn.Linear(num_ftrs, 2)

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))





# In[40]:





model = CnnModel2()
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

test_dl = DeviceDataLoader(test_dl, device)

to_device(model, device);



model = to_device(CnnModel2(), device)



for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    break
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
history = [evaluate(model, val_dl)]

history
epochs = 5

max_lr = 5e-5

grad_clip = 0.2

weight_decay = 0.005

opt_func = torch.optim.Adam

print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)
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
def predict_image(img, model):

    xb = to_device(img.unsqueeze(0), device)

    yb = model(xb)

    _, preds  = torch.max(yb, dim=1)

    return preds[0].item()
def plot_lrs(history):

    lrs = np.concatenate([x.get('lrs', []) for x in history])

    plt.plot(lrs)

    plt.xlabel('Batch no.')

    plt.ylabel('Learning rate')

    plt.title('Learning Rate vs. Batch no.');



plot_lrs(history)
img, label = test_df[0]

plt.imshow(img[0], cmap='gray')

print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])
img, label = test_df[10]

plt.imshow(img[0], cmap='gray')

print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])
img, label = test_df[69]

plt.imshow(img[0], cmap='gray')

print('Label:', dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])
evaluate(model, val_dl)
evaluate(model, test_dl)