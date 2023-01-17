# Data from Chest X-Ray (Pneumonia) Dataset on Kaggle.com
# For Google Colab
# !pip install git+https://github.com/Kaggle/kaggle-api.git --upgrade
# import os
# credentials = {"username":"allenkong","key":"ff55d7dfb506afea9e36c7aadfaa00e5"}
# os.environ['KAGGLE_USERNAME']=credentials["username"]
# os.environ['KAGGLE_KEY']=credentials["key"]
# For Google Colab
# !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# !unzip chest-xray-pneumonia.zip
#!conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y
#!pip install matplotlib --upgrade --quiet
from pathlib import Path
import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import random
%matplotlib inline
DATA_DIR = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'
OUTPUT_DIR = '../output/kaggle/working'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
VAL_DIR = DATA_DIR + '/test'                             # Contains validation images

label_dir = '../input/coronahack-chest-xraydataset'
label_filename = 'Chest_xray_Corona_Metadata.csv'
target = ['Normal', 'Pnemonia']
df = pd.read_csv(label_dir + '/' + label_filename,names=['idx', 'image_name', 'label','type', 'cat1', 'cat2'], header=0).drop(columns=['idx']).fillna('')
train_df = df[df['type']=='TRAIN'].reset_index(drop=True)
val_df = df[df['type']=='TEST'].reset_index(drop=True)
class My_Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_filename = row['image_name']
        
        img_label = torch.tensor(0, dtype=torch.int64) if row['label'] == target[0] else torch.tensor(1, dtype=torch.int64)
        
        img_fullname = self.root_dir + "/" + str(img_filename)

        img = Image.open(img_fullname).convert('RGB')
        if self.transform:
            img = self.transform(img) 
        return img, img_label
train_tfms = transforms.Compose([transforms.Resize((173, 385)), transforms.RandomCrop((160, 360), padding=2, padding_mode='reflect'), transforms.RandomHorizontalFlip(), 
                                 transforms.ToTensor()])
valid_tfms = transforms.Compose([transforms.Resize((173, 385)), transforms.ToTensor()])

train_dataset = My_Dataset(train_df, TRAIN_DIR, transform=train_tfms)
val_dataset = My_Dataset(val_df, VAL_DIR, transform=valid_tfms)
batch_size = 150
# PyTorch data loaders
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size*2, shuffle=False, num_workers=3, pin_memory=True)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(24, 24))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:32], nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl)
show_batch(val_dl)
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
torch.cuda.is_available()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class PneumoniaClassificationBase(nn.Module):
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
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_score']))
class PneumoniaResnet(PneumoniaClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Dropout(0.5)
        self.network.fc = nn.Linear(num_ftrs, 2)
    
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
model = to_device(PneumoniaResnet(), device)
evaluate(model, val_dl)
model.freeze()
epochs = 15
max_lr = 30e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time
history = []
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
model.unfreeze()
%%time
history += fit_one_cycle(epochs, 1e-5, model, train_dl, val_dl, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)
def plot_accuracies(history):
    accuracies = [x['val_score'] for x in history]
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
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');
plot_lrs(history)
test_dataset = PneumoniaDataset(TEST_DATA, transform=val_transform)
test_dl = DataLoader(test_dataset, batch_size, num_workers=3, pin_memory=True)
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()
img, label = test_dataset[2]
plt.imshow(img[0], cmap='gray')
print('Label:', labels[label], ', Predicted:', labels[predict_image(img, model)])
img, label = test_dataset[7]
plt.imshow(img[0], cmap='gray')
print('Label:', labels[label], ', Predicted:', labels[predict_image(img, model)])
test_dl = DeviceDataLoader(test_dl, device)
evaluate(model, test_dl)
# CNN Model 13 layers with Batch Normalization and Dropout
class CNNModel13(PneumoniaClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(), 
            nn.Linear(512*256, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True))
        
    def forward(self, xb):
        return self.network(xb)
model2 = to_device(CNNModel13(), device)
history2 = [evaluate(model2, val_dl)]
history2
%%time
history2 += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
plot_accuracies(history2)
plot_losses(history2)
evaluate(model2, test_dl)
# CNN Model 8 layers
class CNNModel8(PneumoniaClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(), 
            nn.Linear(262144, 32),
            nn.ReLU(),
            nn.Linear(32, 2))
        
    def forward(self, xb):
        return self.network(xb)
model3 = to_device(CNNModel9(), device)
history3 = [evaluate(model3, val_dl)]
history3
%%time
history3 += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
plot_accuracies(history3)
plot_losses(history3)
evaluate(model3, test_dl)
def predict_single(image):
    #print(image.shape)
    # add one dimension at position 0. This four dimension shape is similar to a batch
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    prediction = model(xb).squeeze()
    return print('prediction: ', target[1] if prediction[1]>prediction[0] else target[0])
for i in range(5):
    j = random.randint(100, 500)

    plt.imshow(val_dataset[j][0].permute(1, 2, 0))
    plt.show()
    
    print("item ",j)
    print("label: ", target[val_dataset[j][1].int()])
    predict_single(val_dataset[j][0])
!pip install jovian --upgrade -q
import jovian
jovian.commit(project='pneumonia-detection-resnet')
