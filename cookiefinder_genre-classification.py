import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.models as models
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import librosa.feature
import librosa.display
import glob
from matplotlib import pyplot as plt
!pip install jovian --upgrade --quiet

import jovian
jovian.reset()
DATA_DIR = '../input/gtzan-dataset-music-genre-classification/Data/'
genres_path = DATA_DIR + 'genres_original/'
images_path = DATA_DIR + 'images_original/'
def display_mfcc(song):
   y, _ = librosa.load(song)
   mfcc = librosa.feature.mfcc(y)

   plt.figure(figsize=(10, 4))
   librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
   plt.colorbar()
   plt.title(song)
   plt.tight_layout()
   plt.show()
display_mfcc(genres_path + 'blues/blues.00000.wav')
display_mfcc(genres_path + 'metal/metal.00000.wav')
def extract_features_song(song):
    y, _ = librosa.load(song)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]
def generate_features_and_labels():
    '''
    Produce MFCC values and genre names 
    from all the songs in the dataset
    '''
    
    # Prepare a list for all the features and all the labels
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Extract features and genre names from each song
    for genre in genres:
        sound_files = glob.glob(genres_path + genre + '/*.wav')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for i, song in enumerate(sound_files):
        
            # Handle corrupt song 
            if 'jazz.00054.wav' in song:
                features = extract_features_song(sound_files[i - 1])
            else:
                features = extract_features_song(song)
                
            all_features.append(features)

    return np.stack(all_features)
features = generate_features_and_labels()
labels = np.zeros(1000)
index = 0
for i in range(1000):
    if i % 100 == 0:
        index += 1
    labels[i] = index
labels -= 1
features = features.astype('float32')
labels = labels.astype('int64')
from torch.utils.data import TensorDataset, DataLoader

features_tensor = torch.from_numpy(features)
labels_tensor = torch.from_numpy(labels)
dataset = TensorDataset(features_tensor, labels_tensor)
from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [800, 200])

len(train_ds), len(val_ds)
batch_size = 32

train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)
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
train_loader = DeviceDataLoader(train_dl, device)
val_loader = DeviceDataLoader(val_dl, device)
for xb, yb in train_loader:
    print(xb.shape)
    print(yb.shape)
    break
img_data = ImageFolder(root = images_path, transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
img_train_ds, img_val_ds = random_split(img_data, [800, 199])
batch_size = 32

img_train_dl = DataLoader(img_train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
img_val_dl = DataLoader(img_val_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break
show_batch(img_train_dl)
img_train_loader = DeviceDataLoader(img_train_dl, device)
img_val_loader = DeviceDataLoader(img_val_dl, device)
for xb, yb in img_train_loader:
    print(xb.shape)
    print(yb.shape)
    break
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class SongClassificationBase(nn.Module):
    
    def training_step(self, batch):
        songs, labels = batch 
        out = self(songs)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        songs, labels = batch 
        out = self(songs)                    # Generate predictions
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
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
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
class LogReg(SongClassificationBase):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(25000, 10)
        
    def forward(self, xb):
        out = self.linear(xb)
        return out
model0 = to_device(LogReg(), device)
epochs = 15
max_lr = 3e-4
opt_func = torch.optim.Adam
grad_clip = 1e-2
weight_decay = 1e-4
%%time
history0 = fit_one_cycle(epochs, max_lr, model0, train_loader, val_loader, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
accuracies = [result['val_acc'] for result in history0]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
torch.save(model0.state_dict(), 'project-LogReg.pth')

jovian.log_hyperparams(arch='LogReg', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)

jovian.log_metrics(val_loss=history0[-1]['val_loss'], 
                   val_acc=history0[-1]['val_acc'],
                   train_loss=history0[-1]['train_loss'])
class FNN(SongClassificationBase):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(25000, 1024),
                        nn.ReLU(inplace = True),
                        nn.Linear(1024, 512),
                        nn.ReLU(inplace = True),
                        nn.Linear(512, 256),
                        nn.ReLU(inplace = True),
                        nn.Linear(256, 128),
                        nn.ReLU(inplace = True),
                        nn.Linear(128, 64),
                        nn.ReLU(inplace = True),
                        nn.Linear(64, 10)
        )
    
    def forward(self, xb):
        out = self.model(xb)
        return out
model1 = to_device(FNN(), device)
history1 = [evaluate(model1, val_loader)]
history1
epochs = 15
max_lr = 3e-4
opt_func = torch.optim.Adam
grad_clip = 1e-2
weight_decay = 1e-4
%%time
history1 += fit_one_cycle(epochs, max_lr, model1, train_loader, val_loader, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
accuracies = [result['val_acc'] for result in history1]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
torch.save(model1.state_dict(), 'project-6lFNN.pth')
jovian.log_hyperparams(arch='6l FNN', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)
jovian.log_metrics(val_loss=history1[-1]['val_loss'], 
                   val_acc=history1[-1]['val_acc'],
                   train_loss=history1[-1]['train_loss'])
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class CNN(SongClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 16 x 16

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 1024 x 8 x 8
            
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 4096 x 4 x 4

            nn.Flatten(), 
            nn.Linear(4096 * 4 * 4, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Linear(64, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)
model2 = to_device(CNN(), device)
model2.apply(weights_init)
history2 = [evaluate(model2, img_val_loader)]
history2
epochs = 30
max_lr = 3e-4
opt_func = torch.optim.Adam
grad_clip = 1e-2
weight_decay = 1e-4
%%time
history2 += fit_one_cycle(epochs, max_lr, model2, img_train_loader, img_val_loader, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
accuracies = [result['val_acc'] for result in history2]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
torch.save(model2.state_dict(), 'project-CNN.pth')

jovian.log_hyperparams(arch='CNN', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)

jovian.log_metrics(val_loss=history2[-1]['val_loss'], 
                   val_acc=history2[-1]['val_acc'],
                   train_loss=history2[-1]['train_loss'])
class ResNet(SongClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return self.network(xb)
    
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
model3 = to_device(ResNet(), device)
model3.apply(weights_init)
history3 = [evaluate(model3, img_val_loader)]
history3
model3.freeze()
%%time
epochs = 10
history3 += fit_one_cycle(epochs, max_lr, model3, img_train_loader, img_val_loader, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
model3.unfreeze()
%%time
history3 += fit_one_cycle(15, max_lr, model3, img_train_loader, img_val_loader, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
torch.save(model2.state_dict(), 'project-Resnet.pth')

jovian.log_hyperparams(arch='Resnet34', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)

jovian.log_metrics(val_loss=history3[-1]['val_loss'], 
                   val_acc=history3[-1]['val_acc'],
                   train_loss=history3[-1]['train_loss'])
jovian.commit(project='genre-classify', environment=None, 
              outputs=['project-LogReg.pth','project-6lFNN.pth','project-CNN.pth','project-Resnet.pth'])
