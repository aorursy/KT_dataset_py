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
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from torchvision.utils import make_grid
%matplotlib inline
train_path = "../input/intel-image-classification/seg_train/seg_train/"
val_path ="../input/intel-image-classification/seg_test/seg_test/"
test_path = "../input/intel-image-classification/seg_pred/"
stats = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
train_tf = tt.Compose([  tt.Resize((150,150)), 
                         tt.RandomHorizontalFlip(),
                         tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tf = tt.Compose([tt.Resize((150,150)),tt.ToTensor(), tt.Normalize(*stats)])
train_ds = ImageFolder(train_path, transform = train_tf)
val_ds = ImageFolder(val_path, transform = valid_tf)
batch_size = 200
torch.manual_seed(42)
train_dl = DataLoader(train_ds,batch_size,shuffle = True, num_workers = 4, pin_memory = True)
valid_dl = DataLoader(val_ds, 300, num_workers = 4, pin_memory =True)
def show_batch(dl):
    for img, _ in dl:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(img[:64], nrow=8).permute(1, 2, 0))
        break
show_batch(train_dl)
classes = train_dl.dataset.classes
def get_default_device():
    """check if gpu is available"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
def to_device(data, device):
    """changing the device"""
    if isinstance(data,(list,tuple)):
        return [to_device(x, device) for x in data]
    else:
        return data.to(device, non_blocking = True)
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
valid_dl = DeviceDataLoader(valid_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs,dim = 1)
    return torch.tensor(torch.sum((preds == labels)).item()/len(preds))
class imgclassificationbase(nn.Module):
    def training_step(self, batch):
        imgs, labs = batch
        out = self(imgs)
        loss = F.cross_entropy(out,labs)
        return loss
    
    def validation_step(self, batch):
        img, labs = batch
        out = self(img)
        loss = F.cross_entropy(out,labs)
        acc = accuracy(out,labs)
        return {"val_loss":loss.detach(), "val_acc":acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
def conv_block(inc,outc,pool = False):
    block = [
        nn.Conv2d(inc,outc,kernel_size=3, padding =1),
        nn.BatchNorm2d(outc),
        nn.ReLU(inplace=True)
    ]
    if pool:
        block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)

class ResNet(imgclassificationbase):
    def __init__(self,inc,nc):
        super().__init__()
        self.conv1 = conv_block(inc, 16)
        self.conv2 = conv_block(16, 32, pool = True)
        self.res1 = nn.Sequential(conv_block(32,32),conv_block(32,32))
        
        self.conv3 = conv_block(32,64,pool = True)
        self.conv4 = conv_block(64,128,pool = True)
        self.res2 = nn.Sequential(conv_block(128,128),conv_block(128,128))
        
        self.conv5 = conv_block(128,256,pool = True)
        self.conv6 = conv_block(256,512,pool = True)
        self.res3 = nn.Sequential(conv_block(512,512),conv_block(512,512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, nc))
        
    def forward(self, ip):
        out = self.conv1(ip)
        out = self.conv2(out)
        out = self.res1(out)+out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out)+out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out)+out
        out = self.classifier(out)
        return out
model= to_device(ResNet(3,6),torch.device("cuda"))
model
def try_batch(dl):
    
    for images, labels in dl:
        
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[41])
        break

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
history = [evaluate(model, valid_dl)]
history
epochs = 2
lr = 1e-3
grad_clip = 1e-1
weight_decay = 1e-5
optim_func = torch.optim.AdamW
history+=fit_one_cycle(epochs,lr,model,train_dl,valid_dl,grad_clip = grad_clip, weight_decay = 1e-5, opt_func = optim_func)
epochs = 4
lr = 1e-4 
weight_decay = 1e-6
history+=fit_one_cycle(epochs,lr,model,train_dl,valid_dl,grad_clip = grad_clip, weight_decay = 1e-5, opt_func = optim_func)
def plot_losses(history):
    losses = [x["val_loss"] for x in history]
    plt.plot(losses,"-x")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss vs epochs")
    
def plot_acc(history):
    accs = [x["val_acc"] for x in history]
    plt.plot(accs,"-x")
    plt.xlabel("epoch")
    plt .ylabel("accuracy")
    plt.title("accuracy vs epochs")
plot_losses(history)
plot_acc(history)
def pred_single(img,model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    return preds[0].item()
    
test_ds = ImageFolder(test_path, transform = valid_tf)
img, _ = test_ds[0]
plt.imshow(img.permute(1,2,0))
pred = pred_single(img, model)
classes[pred]
img, _ = test_ds[25]
pred = pred_single(img, model)
print("prediction:\t",classes[pred])
plt.imshow(img.permute(1,2,0))
img , _ = test_ds[40]
pred = pred_single(img,model)
print("prediction:\t",classes[pred])
plt.imshow(img.permute(1,2,0))
img, _ = test_ds[5234]
pred = pred_single(img, model)
print("prediction:\t",classes[pred])
plt.imshow(img.permute(1,2,0))
img, _ = test_ds[7200]
pred = pred_single(img, model)
print("prediction:\t",classes[pred])
plt.imshow(img.permute(1,2,0))
img, _ = test_ds[2320]
pred = pred_single(img, model)
print("prediction:\t",classes[pred])
plt.imshow(img.permute(1,2,0))
projectname = "intelproj"
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project = projectname,environment=None)
