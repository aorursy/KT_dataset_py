import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



from PIL import Image



import torch

from torch import optim

import torch.nn as nn

from torchvision import transforms, datasets, models

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader

from torchvision.utils import make_grid

from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor

import torch.nn.functional as F



from tqdm.notebook import tqdm
DATA_DIR = '../input/Multi-class Weather Dataset'

print(os.listdir(DATA_DIR))

torch.manual_seed(42)

labels = {0: 'Cloudy',1: 'Rain',2:'Shine',3:'Sunrise'}
class MultiClassWeatherDataset(Dataset):

    def __init__(self, path,transform=None):

        self.path = path

        self.transform = transform

        self.all_images = ImageFolder(self.path,transform=self.transform)



    def __len__(self):

        return len(self.all_images) 

    

    def __getitem__(self, idx):

        all_img,all_label = self.all_images[idx]

        return all_img,all_label
basic_transform = transforms.Compose([transforms.Resize((256,256)),

                                       transforms.ToTensor()])



adv_transform = transforms.Compose([transforms.Resize((256,256),interpolation=Image.NEAREST),

                                   transforms.ColorJitter(0.05,0.02,0.01,0.01),

                                   transforms.RandomHorizontalFlip(0.2),

                                   transforms.RandomVerticalFlip(0.2),

                                   transforms.RandomRotation(10),

                                   transforms.ToTensor(),

                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = MultiClassWeatherDataset(DATA_DIR, transform=basic_transform)

len(dataset)
def decode_target(target, text_labels=False):

    result = []

    for i, x in enumerate(target):

        if text_labels:

            result.append(labels[i] + "(" + str(i) + ")")

        else:

            result.append(str(i))

    return ' '.join(result)
def show_sample(img, target, invert=True):

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print(f'Label: {decode_target(target, text_labels=True)}')
dataset[0][0].shape
img,labl = dataset[0]

show_sample(img,[labl],invert=False)
val_pct = 0.2

test_pct = 0.2

val_size = int(val_pct * len(dataset))

test_size = int(test_pct * len(dataset))

train_size = len(dataset)- (val_size+test_size)

batch_size = 128



train_ds, val_ds ,test_ds = random_split(dataset, [train_size, val_size,test_size])

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)

print(f'Train Size:{train_size}\nValidation Size:{val_size}\nTest Size:{test_size}')
def show_batch(dl, invert=True):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = 1-images if invert else images

        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))

        break



def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
show_batch(train_dl)
class WeatherBase(nn.Module):

    def training_step(self, batch):

        images, targets = batch 

        out = self(images)                      

        loss = F.cross_entropy(out, targets)      

        return loss

    

    def validation_step(self, batch):

        images, targets = batch

        # Generate predictions

        out = self(images)

        # Calculate loss 

        loss = F.cross_entropy(out, targets)  

        score = accuracy(out, targets)

        return {'val_loss': loss.detach(), 'val_score': score }

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        # Combine losses

        epoch_loss = torch.stack(batch_losses).mean()   

        batch_scores = [x['val_score'] for x in outputs]

        # Combine accuracies

        epoch_score = torch.stack(batch_scores).mean()      

        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_score']))

    

class WeatheCnnModel(WeatherBase):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Dropout2d(0.2),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),



            nn.AdaptiveAvgPool2d(1),



            nn.Flatten(), 

            nn.Linear(64, 512),

            nn.ReLU(),

            nn.Linear(512, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 4),

            nn.Sigmoid()

        )

        

    def forward(self, xb):

        return self.network(xb)
model = WeatheCnnModel()

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
def try_batch(dl):

    for images, labels in dl:

        print('images.shape:', images.shape)

        out = model(images)

        print('out.shape:', out.shape)

        print('out[0]:', out[0])

        break



try_batch(train_dl)
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

        for batch in tqdm(train_loader):

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
model = to_device(WeatheCnnModel(), device)

evaluate(model, val_dl)
num_epochs = 5

opt_func = torch.optim.Adam

lr= 0.001
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
def plot_losses(history):

    train_losses = [x.get('train_loss') for x in history]

    val_losses = [x['val_loss'] for x in history]

    plt.plot(train_losses, '-bx')

    plt.plot(val_losses, '-rx')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['Training', 'Validation'])

    plt.title('Loss vs. No. of epochs');



def plot_accuracies(history):

    accuracies = [x['val_score'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');



def predict_single(image):

    xb = image[0].unsqueeze(0)

    xb = to_device(xb, device)

    preds = model(xb)

    prediction = preds[0]

    max_probs,labl = (torch.max(prediction,dim=0))

    actual_label = labels[image[1]] + "(" + str(image[1]) + ")"

    print("Prediction: ", labl)

    print("Probability: ", max_probs)

    show_sample(image[0], prediction)

    print("Actual_label:{0}".format(actual_label))
img, target =  test_ds[0]

img.shape
predict_single(test_ds[100])
plot_losses(history)
plot_accuracies(history)