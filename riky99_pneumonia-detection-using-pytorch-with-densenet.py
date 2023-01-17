!pip install jcopdl

!pip install jcopml
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# COMMON PACKAGES

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import torch

from torch import nn, optim

from jcopdl.callback import Callback, set_config



import helper



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# DATASET & DATALOADER

from torchvision import datasets, transforms

from torch.utils.data import DataLoader
bs = 32

crop_size = 224



train_transform = transforms.Compose([

    transforms.RandomRotation(10), #rotation 10%

    transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)), #max zoom 70% from data

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

])



test_transform = transforms.Compose([ 

    transforms.Resize(230), #256 replaced 230 so that the size is not far from CenterCrop 224

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

]) #rule



train_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/train", transform=train_transform) 

trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)



test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/val", transform=test_transform)

testloader = DataLoader(test_set, batch_size=bs, shuffle=True)
len(train_set.classes)
feature, target = next(iter(trainloader))

feature.shape
#label category

label2cat = train_set.classes

label2cat
#how to use Pretrained-Models

from torchvision.models import densenet161



mnet = densenet161(pretrained=True) #True: download model & weightnya 



#freze weight

for param in mnet.parameters():

    param.requires_grad = False
mnet
#replacing classifier sequential

mnet.classifier = nn.Sequential(

    nn.Linear(2208, 2), #2 total class

    nn.LogSoftmax() 

)
#custom arsitektur

class Customdensenet161(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        self.mnet = densenet161(pretrained=True) #arsitektur

        self.freeze()

        self.mnet.classifier = nn.Sequential(  

            #linear_block(2208, 1, activation="lsoftmax")

            nn.Linear(2208, output_size),

            nn.LogSoftmax()

        )

        

    def forward(self, x):

        return self.mnet(x)



    def freeze(self):

        for param in self.mnet.parameters():

            param.requires_grad = False

            

    def unfreeze(self):        

        for param in self.mnet.parameters():

            param.requires_grad = True        
config = set_config({

    "output_size": len(train_set.classes),

    "batch_size": bs,

    "crop_size": crop_size

})
model = Customdensenet161(config.output_size).to(device)

criterion = nn.NLLLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)

callback = Callback(model, config, early_stop_patience=2, outdir="model")
#training 

from tqdm.auto import tqdm



def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):

    if mode == "train":

        model.train()

    elif mode == "test":

        model.eval()

    cost = correct = 0

    for feature, target in tqdm(dataloader, desc=mode.title()):

        feature, target = feature.to(device), target.to(device)

        output = model(feature)

        loss = criterion(output, target)

        

        if mode == "train":

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        

        cost += loss.item() * feature.shape[0]

        correct += (output.argmax(1) == target).sum().item()

    cost = cost / len(dataset)

    acc = correct / len(dataset)

    return cost, acc
#training standart

while True:

    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)

    with torch.no_grad():

        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

    

    # Logging

    callback.log(train_cost, test_cost, train_score, test_score)



    # Checkpoint

    callback.save_checkpoint()

        

    # Runtime Plotting

    callback.cost_runtime_plotting()

    callback.score_runtime_plotting()

    

    # Early Stopping

    if callback.early_stopping(model, monitor="test_score"):

        callback.plot_cost()

        callback.plot_score()

        break
model.unfreeze()

optimizer = optim.AdamW(model.parameters(), lr=1e-5)



callback.reset_early_stop()

callback.early_stop_patience = 5
#training standart 



while True:

    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)

    with torch.no_grad():

        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

    

    # Logging

    callback.log(train_cost, test_cost, train_score, test_score)



    # Checkpoint

    callback.save_checkpoint()

        

    # Runtime Plotting

    callback.cost_runtime_plotting()

    callback.score_runtime_plotting()

    

    # Early Stopping

    if callback.early_stopping(model, monitor="test_score"):

        callback.plot_cost()

        callback.plot_score()

        break



test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test/",transform=test_transform)

testloader = DataLoader(test_set,batch_size = bs)



with torch.no_grad():

        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

        print(f"Test accuracy:{test_score}")

feature, target = next(iter(testloader))

feature, target = feature.to(device), target.to(device)
with torch.no_grad():

    model.eval()

    output = model(feature)

    preds = output.argmax(1)

preds
fig, axes = plt.subplots(6, 6, figsize=(24, 24))

for image, label, pred, ax in zip(feature, target, preds, axes.flatten()):

    ax.imshow(image.permute(1, 2, 0).cpu())

    font = {"color": 'r'} if label != pred else {"color": 'g'}        

    label, pred = label2cat[label.item()], label2cat[pred.item()]

    ax.set_title(f"L: {label} | P: {pred}", fontdict=font);

    ax.axis('off');