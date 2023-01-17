import numpy as np
import pandas as pd
import os
import glob

from PIL import Image

import torch 
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans
from torchvision import models
train_images = list(glob.iglob('../input/train/*.jpg'))
transforms = trans.Compose([
    trans.RandomResizedCrop(224),
    trans.ToTensor()
])

class DogCatDataset:
    def __init__(self, images, train=True, transform=None):
        df = pd.DataFrame({'path': images})
        if train:
            df['label'] = df.path.map(lambda x: x.split('/')[-1].split('.')[0])
            df['id'] = df.path.map(lambda x: x.split('/')[-1].split('.')[1])
        else:
            df['id'] = df.path.map(lambda x: x.split('/')[-1].split('.')[0])

        self.train = train
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img = Image.open(self.df.at[i, 'path'])
        
        if self.train:
            y = 1 if self.df.at[i, 'label'] == 'dog' else 0
        else:
            y = self.df.at[i, 'id'].astype('str')
        
        if self.transform:
            img = self.transform(img)
        return img, y
train_dataset = DogCatDataset(train_images, transform=transforms)
train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
input_size = 224
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
def train(e):
    model.train()
    l = a = n = 0
    for i, (X, Y) in enumerate(train_dl):
        optimizer.zero_grad()
        X = X.to(device)
        Y = Y.to(device)
        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()
        optimizer.step()
        
        l += loss.item()
        a += (preds.argmax(dim=1) == Y).sum().item() / Y.shape[0]
        n += 1
        if (i+1) % 100 == 0:
            print(f"Epoch {e} Iter {i+1} Loss {l/n} Accuracy {a/n}")
            l = n = a = 0
for i in range(5):
    train(5)