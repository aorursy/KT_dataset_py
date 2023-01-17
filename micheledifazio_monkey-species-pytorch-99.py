import os
import glob
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as tt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/10-monkey-species/monkey_labels.txt")
label_dictionary = {
    
    "n0":"mantled_howler",
    "n1":"patas_monkey",
    "n2":"bald_uakari",
    "n3":"japanese_macaque",
    "n4":"pygmy_marmoset",
    "n5":"white_headed_capuchin",
    "n6":"silvery_marmoset",
    "n7":"common_squirrel_monkey",
    "n8":"black_headed_night_monkey",	
    "n9":"nilgiri_langur"
}


categories = {
    
    "mantled_howler":0,
    "patas_monkey":1,
    "bald_uakari":2,
    "japanese_macaque":3,
    "pygmy_marmoset":4,
    "white_headed_capuchin":5,
    "silvery_marmoset":6,
    "common_squirrel_monkey":7,
    "black_headed_night_monkey":8,
    "nilgiri_langur":9
}
def data_dir(phase):
    root_dir = "../input/10-monkey-species/"
    target_path = os.path.join(root_dir + phase + "/**/**/*.jpg")
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

train_list = data_dir(phase="training")
def connect_names(directory):
    for index in range(len(directory)):
        split_path = directory[index].split("/")
        get_index = split_path[-2]
        get_name = label_dictionary[get_index]
        get_name_value = categories[get_name]
        
new_dir = connect_names(train_list)
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.root_dir)
    
    def __getitem__(self, index):
        
        image_path = self.root_dir[index]
        image = default_loader(image_path)
        
        split = image_path.split("/")
        get_index = split[-2]
        get_label = label_dictionary[get_index]
        label_value = categories[get_label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_value
dataset = ds
for idx, (data, image) in enumerate(dataset):
    print(idx)
transform_ds = tt.Compose([
    tt.Resize((224,244)),
    tt.RandomHorizontalFlip(),
    tt.ToTensor()
])

ds = MyDataset(root_dir=train_list, transform=transform_ds)
val_ds_size = int(len(ds) * 0.1)
train_ds_size = len(ds) - val_ds_size
train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
batch_size= 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=3, pin_memory=True)
def show_batch(train_dl):
    for images, _ in train_dl:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:32], nrow=8).permute(1,2,0))
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
    
    def epoch_end(self, epoch, epochs, result):
        print("Epoch: [{}/ {}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch+1, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 10)
        
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
    
def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    
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
        model.epoch_end(epoch, epochs, result)
        history.append(result)
    return history
model = to_device(ResNet(), device)
history = [evaluate(model, val_dl)]
history
model.freeze()
epochs = 5
max_lr = 10e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        weight_decay=weight_decay, grad_clip=grad_clip,
                        opt_func=opt_func)
model.unfreeze()
epochs = 5
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        weight_decay=weight_decay, grad_clip=grad_clip,
                        opt_func=opt_func)
def test_dir(phase):
    root_dir = "../input/10-monkey-species/"
    test_dir = os.path.join(root_dir+phase+"/**/**/*.jpg")
    path_list = []
    for path in glob.glob(test_dir):
        path_list.append(path)
    return path_list
test_directory = test_dir(phase="validation")   
transform_test = tt.Compose([
    tt.Resize((224,224)),
    tt.ToTensor()
])

test_ds = MyDataset(root_dir=test_directory, transform=transform_test)
test_dl = DataLoader(test_ds, batch_size, num_workers=3, pin_memory=True)
test_dl = DeviceDataLoader(test_dl, device)
def prediction(images, model):
    xb = to_device(images.unsqueeze(0), device)
    out = model(xb)
    _, preds = torch.max(out, dim=1)
    prediction = list(categories)[preds[0].item()]
    return prediction
images, labels = test_ds[10]
print("Label:", list(categories)[labels])
print("Prediction:", prediction(images, model))
plt.imshow(images.permute(1,2,0))
images, labels = test_ds[200]
print("Label:", list(categories)[labels])
print("Prediction:", prediction(images, model))
plt.imshow(images.permute(1,2,0))
def plot_loss(history):
    train_loss = [x.get("train_loss") for x in history]
    val_loss = [x["val_loss"] for x in history]
    plt.plot(train_loss, "-rx")
    plt.plot(val_loss, "-bx")
    plt.legend(["Train loss", "Validation loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs number of Epochs")

plot_loss(history)
def plot_accuracy(history):
    accuracy = [x["val_acc"] for x in history]
    plt.plot(accuracy, "-bx")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs number of Epochs")

plot_accuracy(history)