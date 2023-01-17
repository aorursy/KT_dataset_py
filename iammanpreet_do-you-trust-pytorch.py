import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import os
Path="../input/chest-xray-pneumonia/chest_xray/"
Classes=['NORMAL','PNEUMONIA']
Batch_size=64
print("total number of pneumonia examples are",len(os.listdir(Path+"train/"+Classes[1])))
print("total number of normal examples are",len(os.listdir(Path+"train/"+Classes[0])))
dataset=ImageFolder(Path+"train",
                   transform=tt.Compose([tt.Resize((255,255)),
                                        tt.ToTensor()]))

dataset.class_to_idx
for i,l in dataset:
    print(i.shape)
    print(l)
    break
%matplotlib inline
def show_example(img,l):
    print("Label is ", dataset.classes[l])
    plt.imshow(img.permute(1,2,0))
    plt.show()
show_example(*dataset[18])
from torch.utils.data.sampler import SubsetRandomSampler
indices=list(range(len(dataset)))
np.random.shuffle(indices)
train_indices, valid_indices = indices[500:], indices[:500]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_dataset = DataLoader(
    dataset,
    batch_size=Batch_size,#args.batch_size,
    num_workers=2,#args.loader_num_workers,
    
    sampler=train_sampler)

val_dataset = DataLoader(
    dataset,
    batch_size=Batch_size,#args.batch_size,
    num_workers=2,#args.loader_num_workers,
    
    sampler=valid_sampler)
len(train_dataset),len(val_dataset)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:60], nrow=10).permute(1, 2, 0))
        break
        
show_batch(train_dataset)
def get_ratios(label):
    N=len(label)
    wp=np.sum(label)/N
    wn=1-wp
    return wp,wn
def getloss():
    def compute_loss(pred,y):
        pos_loss=-1*(torch.mean(y*torch.log(pred+1e-7)))
        neg_loss=-1*(torch.mean((1-y)*torch.log(1-pred+1e-7)))
        return pos_loss+neg_loss
    return compute_loss
base_model = Net()
print(base_model)

base_model=models.densenet121(pretrained=True)
base_model
for params in base_model.parameters():
    params.requires_grad=True
base_model.classifier=nn.Sequential(
    nn.Linear(1024,512),
    nn.ReLU(inplace=True),
    nn.Linear(512,512),
    nn.ReLU(inplace=True),
    nn.Linear(512,1),
    nn.Sigmoid()
)
wp,wn=get_ratios(dataset.targets)
wp,wn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12= nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
        self.dropout = nn.Dropout(0.50)
#         self.avgpool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(F.relu(self.conv10(x)))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(F.relu(self.conv13(x)))
#         x = self.avgpool(x)   
                                 
        # flatten image input
        x = x.view(-1, 512*7*7)
        
        
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
         # add 3rd hidden layer
        x = self.fc3(x)
        return x

base_model = Net()
print(base_model)
crit=nn.BCELoss()
# crit = compute_loss()
optimizer=torch.optim.Adam(base_model.parameters())

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
base_model=base_model.to(device)
def train(model, epoch, optimizer, train_loader, criterion):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % 25 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_loss / total_size))

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
for epoch in range(1, 5):
        train(base_model, epoch, optimizer, train_dataset,crit)
        test(base_model, val_dataset,crit)
from sklearn.metrics import precision_recall_fscore_support

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'
    
print(device)
def prf1score(model, test_loader):
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
    print(f"precision recall fscore  { precision_recall_fscore_support(target, pred.to(device), average='weighted')}")
    
prf1score(base_model, val_dataset)
from sklearn.metrics import precision_score
def precision(model, test_loader):
    precision=0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
            precision+=precision_score(target.data.view_as(pred).cpu(),pred.cpu())
    print(precision)
    
precision(base_model,val_dataset)