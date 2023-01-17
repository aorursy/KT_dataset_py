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



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# DATASET & DATALOADER

from torchvision import datasets, transforms

from torch.utils.data import DataLoader
bs = 64

crop_size = 224



train_transform = transforms.Compose([

    transforms.RandomRotation(10),

    transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)),

    transforms.ColorJitter(brightness=0.3),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



test_transform = transforms.Compose([

    transforms.Resize(230),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



train_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/train", transform=train_transform)

trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)



test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/val", transform=test_transform)

testloader = DataLoader(test_set, batch_size=bs, shuffle=True)
len(train_set.classes)
feature, target = next(iter(trainloader))

feature.shape
# label category

label2cat = train_set.classes

label2cat
from torchvision.models import mobilenet_v2, densenet121

from jcopdl.layers import linear_block



mnet = mobilenet_v2(pretrained=True)

dnet = densenet121(pretrained=True)



# freeze model

for param in mnet.parameters():

    param.requires_grad = False
# mnet

# dnet
# replacing the final activation with logsoftmax

mnet.classifier = nn.Sequential(

#     linear_block(1280, 2, activation="lsoftmax")

    nn.Linear(1280, 2),

    nn.LogSoftmax()

)



dnet.classifier = nn.Sequential(

    nn.Linear(1280, 2),

    nn.LogSoftmax()

)
# mnet

# dnet
class CustomMobilenetV2(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        self.mnet = mobilenet_v2(pretrained=True)

        self.freeze()

        self.mnet.classifier = nn.Sequential(

#             linear_block(1280, 1, activation="lsoftmax")

            nn.Linear(1280, output_size),

            nn.LogSoftmax(dim=1)

        )

        

    def forward(self, x):

        return self.mnet(x)



    def freeze(self):

        for param in self.mnet.parameters():

            param.requires_grad = False

            

    def unfreeze(self):        

        for param in self.mnet.parameters():

            param.requires_grad = True  

# class CNN(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.conv = nn.Sequential(

#             conv_block(3, 8), #224x224

#             conv_block(8, 16), #112x112

#             conv_block(16, 32), #56x56

#             conv_block(32, 64), #28x28

#             nn.Flatten()

#         )

        

#         self.fc = nn.Sequential(

#             linear_block(12544, 256, dropout=0.1),

#             linear_block(256, 2, activation="lsoftmax")

#         )

        

#     def forward(self, x):

#         x = self.conv(x)

#         x = self.fc(x)

#         return x        





# class CustomDensenet121(nn.Module):

#     def __init__(self, output_size):

#         super().__init__()

#         self.dnet = densenet121(pretrained=True)

#         self.freeze()

#         dnet.classifier = nn.Linear(1280, output_size)

        

#     def forward(self, x):

#         return self.dnet(x)     

    

#     def freeze(self):

#         for param in self.dnet.parameters():

#             param.requires_grad = True

    

#     def unfreeze(self):

#         for param in self.dnet.parameters():

#             param.requires_grad = False

# class CustomDensenet121(nn.Module):

#     def __init__(self, output_size):

#         super().__init__()

#         self.dnet = densenet121(pretrained=True)

#         self.freeze()

#         self.dnet.classifier = nn.Sequential(

# #             linear_block(1280, 1, activation="lsoftmax")

#             nn.Linear(1280, output_size),

#             nn.LogSoftmax(dim=1)

#         )

        

#     def forward(self, x):

#         return self.dnet(x)



#     def freeze(self):

#         for param in self.dnet.parameters():

#             param.requires_grad = False

            

#     def unfreeze(self):        

#         for param in self.dnet.parameters():

#             param.requires_grad = True  
config = set_config({

    "output_size": len(train_set.classes),

    "batch_size": bs,

    "crop_size": crop_size

})
model = CustomMobilenetV2(config.output_size).to(device)

criterion = nn.NLLLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)

callback = Callback(model, config, early_stop_patience=2, outdir="model")
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
from jcopml.utils import save_model
save_model(model, "xray_chest_pneumonia_mobilenet121_v1.pkl")
feature, target = next(iter(testloader))

feature, target = feature.to(device), target.to(device)
with torch.no_grad():

    model.eval()

    output = model(feature)

    preds = output.argmax(1)

preds
from matplotlib import pyplot as plt
fig, axes = plt.subplots(6, 6, figsize=(24, 24))

for image, label, pred, ax in zip(feature, target, preds, axes.flatten()):

    ax.imshow(image.permute(1, 2, 0).cpu())

    font = {"color": 'r'} if label != pred else {"color": 'g'}        

    label, pred = label2cat[label.item()], label2cat[pred.item()]

    ax.set_title(f"L: {label} | P: {pred}", fontdict=font);

    ax.axis('off');
test_set_final = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test", transform=test_transform)

testloader_final = DataLoader(test_set_final, batch_size=bs)



with torch.no_grad():

    test_cost_final, test_score_final = loop_fn("test", test_set_final, testloader_final, model, criterion, optimizer, device)

    print(f"Test Accuracy: {test_score_final}")