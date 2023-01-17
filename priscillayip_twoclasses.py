# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm
%matplotlib inline
!pip install jovian --upgrade -q
import jovian
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
DATA_DIR = '../input/covid19-detection-xray-dataset'
OUTPUT_DIR = '../output/kaggle/working'

TRAIN_DIR = DATA_DIR + '/TrainData'                           # Contains training images
VAL_DIR = DATA_DIR + '/ValData'                             # Contains validation images
TEST_DIR = DATA_DIR + '/NonAugmentedTrain'                  # Contains test images 
subfolder = ['ViralPneumonia', 'COVID-19', 'Normal', 'BacterialPneumonia', 'OversampledAugmentedCOVID-19/COVID-19']
dataset = ImageFolder(TRAIN_DIR, transform=ToTensor())
img, label = dataset[999]
print(img.shape, label)
print(img)
plt.imshow(img.permute(1, 2, 0))
def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
data = []
for folder in os.listdir(TRAIN_DIR):
    # if not augmented
    if folder != subfolder[4].split('/')[0]:
        for file in os.listdir(TRAIN_DIR + "/" + folder):
            data.append((folder, file))
    else:
        for file in sorted(os.listdir(TRAIN_DIR + "/" + subfolder[4])):
            data.append((subfolder[4], file))
    
train_label = pd.DataFrame(data, columns=['Folder', 'File'])
train_label['Class'] = 1
train_label.loc[train_label['Folder']==subfolder[0],'Class'] = 0
train_label.loc[train_label['Folder']==subfolder[2],'Class'] = 2
train_label.loc[train_label['Folder']==subfolder[3],'Class'] = 3
train_label['Class'] = train_label['Class'].astype(int)
train_label.to_csv('train_label.csv')
print(train_label['Class'].value_counts())
print(train_label.head())
plt.hist(train_label['Class'], bins=range(5))
bins_labels(range(5), fontsize=14)
plt.show()
data = []
for folder in os.listdir(VAL_DIR):
    for file in os.listdir(VAL_DIR + "/" + folder):
        data.append((folder, file))
    
val_label = pd.DataFrame(data, columns=['Folder', 'File'])
val_label.loc[val_label['Folder']==subfolder[0],'Class'] = 0
val_label.loc[val_label['Folder']==subfolder[1],'Class'] = 1
val_label.loc[val_label['Folder']==subfolder[2],'Class'] = 2
val_label.loc[val_label['Folder']==subfolder[3],'Class'] = 3
val_label['Class'] = val_label['Class'].astype(int)
val_label.to_csv('val_label.csv')
val_label['Class'].value_counts()
val_label.head()
data = []
for folder in os.listdir(TEST_DIR):
    for file in os.listdir(TEST_DIR + "/" + folder):
        data.append((folder, file))
    
test_label = pd.DataFrame(data, columns=['Folder', 'File'])
test_label.loc[test_label['Folder']==subfolder[0],'Class'] = 0
test_label.loc[test_label['Folder']==subfolder[1],'Class'] = 1
test_label.loc[test_label['Folder']==subfolder[2],'Class'] = 2
test_label.loc[test_label['Folder']==subfolder[3],'Class'] = 3
test_label['Class'] = test_label['Class'].astype(int)
test_label.to_csv('test_label.csv')
test_label['Class'].value_counts()
class My_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_folder, img_filename, img_label = row['Folder'], row['File'], row['Class']
        img_fullname = self.root_dir + "/" + str(img_folder) + "/" + str(img_filename)
        # I don't know why the output image has number of channel = 1. So, I have to convert it to RGB
        img = Image.open(img_fullname).convert('RGB')
        if self.transform:
            img = self.transform(img) 
        return img, img_label
train_tfms = transforms.Compose([transforms.Resize((300,400)), transforms.RandomCrop((288, 384), padding=2, padding_mode='reflect'), transforms.RandomHorizontalFlip(), 
                                 transforms.ToTensor()])
valid_tfms = transforms.Compose([transforms.Resize((300,400)), transforms.ToTensor()])

train_ds = My_Dataset('train_label.csv', TRAIN_DIR, transform=train_tfms)
print(len(train_ds))
print(train_ds[0][0].shape)
train_ds[999]
def show_sample(img, target):
    plt.imshow(img.permute(1, 2, 0))
    print('Labels:', target)
show_sample(*train_ds[999])
val_ds = My_Dataset('val_label.csv', VAL_DIR, transform=valid_tfms)
#test_ds = My_Dataset('test_label.csv', TEST_DIR, transform=valid_tfms)
batch_size=90
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
#test_dl = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(24, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=12).permute(1, 2, 0))
        print(labels)
        break
        
show_batch(train_dl)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    #print("preds: ", preds)
    #print("labels: ", labels)
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
        #print("labels", labels)
        #print("out", out)
        #F2 = F2_score(out, labels)
        #print("passed")
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
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
                                                            #batch x 3 x 288, 384
        self.conv1 = conv_block(in_channels, 64, pool=True)            #batch x 64 x 144 x 192
        self.conv2 = conv_block(64, 128, pool=True)         #batch x 128 x 72 x 96
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))       #batch x 128 x 72 x 96
        
        self.conv3 = conv_block(128, 128, pool=True)        #batch x 128 x 36 x 48
        #self.conv4 = conv_block(256, 256, pool=True)        #batch x 256 x 18 x 24
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))       #batch x 128 x 36 x 48
        #self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))       #batch x 512 x 72 x 88
        
        self.classifier = nn.Sequential(nn.MaxPool2d((12, 12)),    #batch x 128 x 3 x 4
                                        nn.Dropout(0.2),
                                        nn.Flatten(),       #batch x 128 x 3 x 4
                                        nn.Linear(128*3*4, num_classes))    #4
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        #out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    # this function move model to device
    """Move tensor(s) to chosen device"""
    # if "data" is of type list or tuple
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

model = to_device(ResNet9(3, 4), device)
model
with torch.cuda.device('cuda:0'):
    torch.cuda.empty_cache()   
def try_batch(dl):
    for images, labels in dl:
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        print('label:', labels[0])
        break

# no training
try_batch(train_dl)
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
model = to_device(ResNet9(3, 4), device)
evaluate(model, val_dl)
epochs = 12
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time
history = []
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
jovian.reset()
jovian.log_hyperparams({
    'num_epochs': epochs,
    'opt_func': opt_func.__name__,
    'batch_size': batch_size,
    'max_lr': max_lr,
    'grad_clip': grad_clip,
    'weight_decay': weight_decay,
})
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
def plot_learning_rate(history):
    lrs = np.concatenate([x.get('lrs',[]) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('learning rates')
    plt.title('learning rates vs. Batch no.');
plot_learning_rate(history)
jovian.log_metrics(train_loss=history[-1]['train_loss'], 
                   val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'])
def predict_single(image):
    #print(image.shape)
    # add one dimension at position 0. This four dimension shape is similar to a batch
    xb = image.unsqueeze(0)
    #print(xb.shape)
    xb = to_device(xb, device)
    prediction = model(xb)
    _, preds = torch.max(prediction, dim=1)
    print("Prediction: ", subfolder[preds])
for i in range(10):
    j = random.randint(100, 500)
    print("item ",j)
    print("label: ", subfolder[val_ds[j][1]])
    predict_single(val_ds[j][0])
torch.save(model.state_dict(), 'ZerotoGANs-project.pth')
model.state_dict()
model2 = ResNet9(3,4)
model2.load_state_dict(torch.load('./ZerotoGANs-project.pth'))
model2.state_dict()
model2.cuda()
evaluate(model2, val_dl)
jovian.commit(project='ZerotoGANS-Project', environment=None, outputs=['ZerotoGANs-project.pth'])