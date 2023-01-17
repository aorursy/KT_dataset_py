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
!pip install jcopdl
import numpy as np

import matplotlib.pyplot as plt
import torch

from torch import nn, optim

from jcopdl.callback import Callback, set_config



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from jcopdl.utils.dataloader  import MultilabelDataset
bs = 64

crop_size = 224



train_transform = transforms.Compose([

    transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)),

     transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



test_transform = transforms.Compose([

    transforms.Resize(230),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



train_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/train/", transform=train_transform)

trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)



test_set =datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/val/", transform=test_transform)

testloader = DataLoader(test_set, batch_size=bs, shuffle=True)
label2cat = train_set.classes

label2cat
mnet.classifier = nn.Sequential(

    nn.Linear(1280, 2),

    nn.Sigmoid()

)
class CustomMobilenetV2(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        self.mnet = mobilenet_v2(pretrained=True)

        self.freeze()

        self.mnet.classifier = nn.Sequential(

            nn.Linear(1280, output_size),

            nn.Sigmoid()

        )

        

    def forward(self, x):

        return self.mnet(x)



    def freeze(self):

        for param in self.mnet.parameters():

            param.requires_grad = False

            

    def unfreeze(self):        

        for param in self.mnet.parameters():

            param.requires_grad = False       
config = set_config({

    "output_size": len(train_set.classes),

    "batch_size": bs,

    "crop_size": crop_size

})
from torchvision.models import mobilenet_v2



mnet = mobilenet_v2(pretrained=True)



for param in mnet.parameters():

    param.requires_grad = True
model = CustomMobilenetV2(config.output_size).to(device)

criterion = nn.BCELoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)

callback = Callback(model, config, early_stop_patience=2, outdir="model")
from tqdm.auto import tqdm



def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):

    if mode == "train":

        model.train()

    elif mode == "test":

        model.eval()

    cost = 0

    for feature, target in tqdm(dataloader, desc=mode.title()):

        feature, target = feature.to(device), target.to(device)

        output = model(feature)

        loss = criterion(output, target)

        

        if mode == "train":

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        

        cost += loss.item() * feature.shape[0]

    cost = cost / len(dataset)

    return cost
while True:

    train_cost = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)

    with torch.no_grad():

        test_cost = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

    

    # Logging

    callback.log(train_cost, test_cost)



    # Checkpoint

    callback.save_checkpoint()

        

    # Runtime Plotting

    callback.cost_runtime_plotting()

    

    # Early Stopping

    if callback.early_stopping(model, monitor="test_cost"):

        callback.plot_cost()

        break
model.unfreeze()

optimizer = optim.AdamW(model.parameters(), lr=1e-5)



callback.reset_early_stop()

callback.early_stop_patience = 5
while True:

    train_cost = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)

    with torch.no_grad():

        test_cost = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

    

    # Logging

    callback.log(train_cost, test_cost)



    # Checkpoint

    callback.save_checkpoint()

        

    # Runtime Plotting

    callback.cost_runtime_plotting()

    

    # Early Stopping

    if callback.early_stopping(model, monitor="test_cost"):

        callback.plot_cost()

        break


#n.b. jcopdl has loaded the best weight when performing early stopping



test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test/", transform=test_transform)

testloader = DataLoader(test_set, batch_size=bs)



with torch.no_grad():

    test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

    print(f"Test accuracy: {test_score}")