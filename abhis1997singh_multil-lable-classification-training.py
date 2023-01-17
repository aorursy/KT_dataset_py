# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/planets-dataset/planet/planet'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/planets-dataset/planet/planet/train_classes.csv")
from pathlib import Path
import torchvision.models as models
import pickle, math
import matplotlib.pyplot as plt
from torch import tensor
import numpy as np
import os
#from torch_lr_finder import LRFinder
import sklearn
from PIL import Image
from torch.utils.data import Dataset

#import pixiedust
import torch
from torch import nn
import pathlib
from torch.utils.data import DataLoader
from torchvision import *
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

transform= transforms.Compose([
    transforms.Resize((124,124)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),              #convert the value to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])         # convert all the value form -1 to 1 for all RGB
])

model =models.resnext50_32x4d(pretrained=True).to(device)
mlb = MultiLabelBinarizer()
a=mlb.fit_transform(df['tags'].str.split()).astype(np.float32)
a
a=torch.from_numpy(a)
a
img_name=df['image_name']
img_name
ext=".jpg"

class planet(Dataset):
    def __init__(self,csv_path,img_dir,ext=".jpg",trans=transform):
        self.df=pd.read_csv(csv_path)
        self.image_dir=img_dir
        self.trans=trans
        
                
        self.img_name=df['image_name']
        
        
        self.label=mlb.fit_transform(df['tags'].str.split()).astype(np.float32)
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        img=Image.open(self.image_dir+self.img_name[idx]+ext)
        img=img.convert('RGB')
        
        label=torch.from_numpy(self.label[idx]).float()
        if self.trans is not None:
            img=self.trans(img)
            
        return img,label
csv="/kaggle/input/planets-dataset/planet/planet/train_classes.csv"
img_dir="/kaggle/input/planets-dataset/planet/planet/train-jpg/"
train=planet(csv,img_dir,transform)
train_set, val_set = torch.utils.data.random_split(train, [38479, 2000])

train_loader = DataLoader(train_set,
                          batch_size=64,
                          shuffle=True)

val_set= DataLoader(train_set,
                          batch_size=64,
                          shuffle=True)


cat=17
cat

for param in model.parameters():
    param.requires_grad=False


model.fc= nn.Sequential(nn.Linear(2048, 500),nn.ReLU(),nn.Linear(500,17),nn.Sigmoid()).to(device)
#y_prob = torch.sigmoid(model.fc)


model.fc.requires_grad=True

loss=nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    tloss=[]
    vloss=[]
    
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
        tot_train=0
        model.train()
#         print(model.training)
        for xb,yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            #print(xb,yb)
            loss = loss_func((model(xb)), yb)
            
            tot_train+=loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        nt=len(train_dl)
        
        
        
        model.eval()
#         print(model.training)
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                #tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        tloss.append(tot_train/nt)
        vloss.append(tot_loss/nv)
        print(epoch,tot_train/nt, tot_loss/nv)
    return tloss, vloss
ltrain,lval = fit(5, model, loss, optimizer, train_loader, val_set)
