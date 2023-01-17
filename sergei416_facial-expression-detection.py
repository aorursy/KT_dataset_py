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
# Data augmentation

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



train_tfms = T.Compose([

    T.RandomCrop(48, padding=8, padding_mode='reflect'),

     #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 

    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

    T.Resize((48, 48)),

    T.RandomHorizontalFlip(), 

    T.RandomRotation(10),

    T.ToTensor(), 

     T.Normalize(*imagenet_stats,inplace=True), 

    #T.RandomErasing(inplace=True)

])



valid_tfms = T.Compose([

     T.Resize((48, 48)), 

    T.ToTensor(), 

     T.Normalize(*imagenet_stats)

])


dataset = ImageFolder(root='/kaggle/input/facial-expression-recognition/train/')



dataset_size = len(dataset)

dataset_size
testdataset = ImageFolder(root='/kaggle/input/facial-expression-recognition/test/', transform = valid_tfms)



testdataset_size = len(testdataset)

testdataset_size
dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
random_image = np.random.randint(0, 28708)

print('Random image number ', random_image)

print('Class label', dict[dataset[random_image][1]])

dataset[random_image][0]
random_image = np.random.randint(0, 28708)

print('Random image number ', random_image)

print('Class label', dict[dataset[random_image][1]])

dataset[random_image][0]
random_image = np.random.randint(0, 28708)

print('Random image number ', random_image)

print('Class label', dict[dataset[random_image][1]])

dataset[random_image][0]
classes = dataset.classes

classes
num_classes = len(dataset.classes)

num_classes
val_size = 1000

train_size = len(dataset) - val_size



train_df, val_df = random_split(dataset, [train_size, val_size])

len(train_df), len(val_df)
val_df.dataset.transform = valid_tfms



train_df.dataset.transform = train_tfms
batch_size = 64



train_dl = DataLoader(train_df, batch_size, shuffle=True, 

                      num_workers=3, pin_memory=True)

val_dl = DataLoader(val_df, batch_size*2, 

                    num_workers=2, pin_memory=True)

test_dl = DataLoader(testdataset, batch_size*2, 

                    num_workers=2, pin_memory=True)
for images, _ in train_dl:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))

    break
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
class CnnModel2(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        # Use a pretrained model

        self.network = models.wide_resnet101_2(pretrained=True)

        # Replace last layer

        num_ftrs = self.network.fc.in_features

        self.network.fc = nn.Linear(num_ftrs, 7)

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))
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
model = to_device(CnnModel2(), device)

train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

test_dl = DeviceDataLoader(test_dl, device)
for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    break
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
# Set search to a larger number to test out more hyperparameters

search = 1
history = [evaluate(model, val_dl)]

history
epochs = np.random.randint(2, 25)

max_lr = np.random.choice([5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6])

grad_clip = np.random.choice([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

weight_decay = np.random.choice([1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5])

opt_func = torch.optim.Adam

print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)
torch.cuda.empty_cache()





history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 

                             grad_clip=grad_clip, 

                             weight_decay=weight_decay, 

                             opt_func=opt_func)
for j in range(search):

    model = to_device(CnnModel2(), device)

    history = [evaluate(model, val_dl)]

    print(history)

    epochs = np.random.randint(2, 25)

    max_lr = np.random.choice([5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6])

    grad_clip = np.random.choice([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    weight_decay = np.random.choice([1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5])

    opt_func = torch.optim.Adam

    print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)

    torch.cuda.empty_cache()





    history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 

                                 grad_clip=grad_clip, 

                                 weight_decay=weight_decay, 

                                 opt_func=opt_func)
for j in range(search):

    model = to_device(CnnModel2(), device)

    history = [evaluate(model, val_dl)]

    print(history)

    epochs = np.random.randint(12,24)

    max_lr = np.random.choice([ 5e-3, 1e-4, 5e-4, 1e-5, 5e-5])

    grad_clip = np.random.choice([0.2, 0.15, 0.1, 0.05])

    weight_decay = np.random.choice([ 5e-3, 1e-4, 5e-4, 1e-5])

    opt_func = torch.optim.Adam

    print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)

    torch.cuda.empty_cache()





    history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 

                                 grad_clip=grad_clip, 

                                 weight_decay=weight_decay, 

                                 opt_func=opt_func)
model = to_device(CnnModel2(), device)

history = [evaluate(model, val_dl)]

print(history)

epochs = 45

max_lr = 0.0001

grad_clip = 0.025

weight_decay = 1e-5

opt_func = torch.optim.Adam

print('epoch = ', epochs, 'lr = ', max_lr, 'grad is ', grad_clip, 'weights = ', weight_decay)

torch.cuda.empty_cache()





history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 

                                 grad_clip=grad_clip, 

                                 weight_decay=weight_decay, 

                                 opt_func=opt_func)
torch.save(model.state_dict(), 'facial.pth')
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
def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');



plot_accuracies(history)
def plot_lrs(history):

    lrs = np.concatenate([x.get('lrs', []) for x in history])

    plt.plot(lrs)

    plt.xlabel('Batch no.')

    plt.ylabel('Learning rate')

    plt.title('Learning Rate vs. Batch no.');



plot_lrs(history)
evaluate(model, val_dl)['val_acc']
evaluate(model, test_dl)['val_acc']
model = to_device(CnnModel2(), device)

model.load_state_dict(torch.load('/kaggle/input/facial/facial.pth'))
def predict_image(img, model):

    xb = to_device(img.unsqueeze(0), device)

    yb = model(xb)

    _, preds  = torch.max(yb, dim=1)

    return preds[0].item()
img, label = testdataset[0]

plt.imshow(img[0], cmap='gray')

print('Label:', testdataset.classes[label], ', Predicted:', testdataset.classes[predict_image(img, model)])
random_image = np.random.randint(0, 7178)

print('Random image number ', random_image)

img, label = testdataset[random_image]

plt.imshow(img[0], cmap='gray')

print('Label:', testdataset.classes[label], ', Predicted:', testdataset.classes[predict_image(img, model)])
my_ims = ImageFolder(root='/kaggle/input/test-faces/', transform = valid_tfms)



my_ims_size = len(my_ims)

my_ims_size
img, label = my_ims[0]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])
img, label = my_ims[1]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])
img, label = my_ims[2]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])
img, label = my_ims[3]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])
img, label = my_ims[4]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])
img, label = my_ims[5]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])
img, label = my_ims[6]

plt.imshow(img[0], cmap='gray')

print('Predicted:', dataset.classes[predict_image(img, model)])