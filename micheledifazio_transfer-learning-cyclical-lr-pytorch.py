import os
import torch
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchvision.datasets.folder import default_loader
TRAIN_DIR = "../input/siim-isic-melanoma-classification/jpeg/train"
train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
train_df.head()
class MyDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.label = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
            
        image_path = os.path.join(self.root_dir, f"{self.label.iloc[idx, 0]}.jpg")
        image_label = self.label.iloc[idx, 7]
        
        image = default_loader(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        return (image, image_label)
transform_train = T.Compose([
    T.Resize((128,128)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    #T.RandomErasing()
])

train_ds = MyDataset(
    root_dir=TRAIN_DIR,
    csv_file="../input/siim-isic-melanoma-classification/train.csv",
    transform=transform_train
)
images, labels = train_ds[10]
print("Label:", labels)
plt.imshow(images.permute(1,2,0))
batch_size=128
val_ds_size= int(len(train_ds) * 0.1)
train_ds_size= len(train_ds) - val_ds_size

train_ds, val_ds = random_split(train_ds, [train_ds_size, val_ds_size])
print("train_ds lenght:", len(train_ds))
print("val_ds lenght:", len(val_ds))
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
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
    
    def epoch_end(self, epoch, epochs, result):
        print("Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))
class resnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 2)
        
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad=False
        for param in self.network.fc.parameters():
            param.requires_grad=True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad=True
model = to_device(resnet(), device)
@torch.no_grad()
def evaluate(model, val_dl):
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
        
def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=0,
                 grad_clip=None, opt_func=torch.optim.Adam):
    
    history = []
    opt = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=epochs,
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
                
            opt.step()
            opt.zero_grad()
            
            lrs.append(get_lr(opt))
            sched.step()
            
        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, epochs, result)
        history.append(result)
    return history
history = [evaluate(model, val_dl)]
history
model.freeze() #freeze all the layers (which are already trained) and train only the last one.
epochs= 5
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 10e-4
opt_func = torch.optim.Adam
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        weight_decay=weight_decay, grad_clip=grad_clip,
                        opt_func=opt_func)
model.unfreeze() #unfreeze all the layers and train for some more
%%time

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        weight_decay=weight_decay, grad_clip=grad_clip,
                        opt_func=opt_func)