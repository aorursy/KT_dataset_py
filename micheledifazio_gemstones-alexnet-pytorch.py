import os
import PIL
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
%matplotlib inline
train_data = "../input/gemstones-images/train"
tsf_ds = tt.Compose([
    tt.Resize((128, 128)),
    tt.RandomHorizontalFlip(),
    tt.VerticalRandomFlip(),
    #tt.RandomCrop(32, padding=4, padding_mode="reflect"),
    tt.ToTensor()
])

ds = torchvision.datasets.ImageFolder(root="../input/gemstones-images/train", transform=tsf_ds)
images, labels = ds[8]
print(images.size())
plt.imshow(images.permute(1,2,0))
print(labels)
val_ds_size = int(len(ds) * 0.2)
train_ds_size = len(ds) - val_ds_size
train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
len(train_ds), len(val_ds)
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=3, pin_memory=True)
def show_batch(train_dl):
    for images, labels in train_dl:
        fig, ax = plt.subplots(figsize=(16,16))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1,2,0))
        break
show_batch(train_dl)
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr {:.6f}, train_loss {:.4f}, val_loss {:.4f}, val_acc {:.4f}".format(
            epoch, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))
        
alexnet = models.alexnet()
alexnet
class Gemsalexnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.alexnet(pretrained=True)
        number_of_features = self.network.classifier[6].in_features
        self.network.classifier[6] = nn.Linear(number_of_features, 87)
        
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.classifier[6].parameters():
            param.require_grad = True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64) #128, 64, 128, 128
        self.conv2 = conv_block(64, 128, pool=True) #128, 128, 64, 64
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True) #128, 256, 32, 32
        self.conv4 = conv_block(256, 512, pool=True) #128, 512, 16, 16
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) #128, 512, 16, 16
        
        self.classifier = nn.Sequential(nn.MaxPool2d(16), nn.Flatten(), nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
model = ResNet9(in_channels=3, num_classes=31)
model
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)
            
    def __len__(self):
        return len(self.dl)
    
device = get_device()
device
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=None, 
                  grad_clip=0, opt_func=torch.optim.Adam):
    
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                               steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        lrs = []
        for batch in tqdm(train_dl):
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
            
        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
model = to_device(Gemsalexnet(), device)
history = [evaluate(model, val_dl)]
history
model.freeze()
epochs = 10
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        grad_clip=grad_clip, weight_decay=weight_decay,
                        opt_func=opt_func)
model.unfreeze()
epochs = 10
max_lr = 0.0005
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        grad_clip=grad_clip, weight_decay=weight_decay,
                        opt_func=opt_func)
def plot_losses(history):
    val_loss = [x["val_loss"] for x in history]
    train_loss = [x.get("train_loss") for x in history]
    plt.plot(val_loss, "-rx")
    plt.plot(train_loss, "-bx")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend(["Validation loss", "Train loss"])
    plt.title("Loss vs number of epochs")
    
plot_losses(history)
def plot_accuracy(history):
    accuracy = [x["val_acc"] for x in history]
    plt.plot(accuracy, "-x")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs number of epochs")
    
plot_accuracy(history)
def prediction(images, model):
    xb = to_device(images.unsqueeze(0), device)
    out = model(xb)
    _,preds = torch.max(out, dim=1)
    prediction = ds.classes[preds[0].item()]
    return prediction
test_ds = torchvision.datasets.ImageFolder(
root = "../input/gemstones-images/test", transform=tt.ToTensor())
images, labels = test_ds[2]
print("Label:", test_ds.classes[labels])
print("Prediction:", prediction(images, model))
plt.imshow(images.permute(1,2,0))
