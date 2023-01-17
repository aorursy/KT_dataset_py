# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

from torch import nn

from torchvision import datasets, models, transforms

import torch.nn.functional as func

import torchvision

from torchvision import transforms

from torch.utils.data import TensorDataset,DataLoader

from torch import optim

from torch import device as dev

from sklearn.metrics import classification_report

import torch.utils.data as tdata

from sklearn import model_selection



np.random.seed(1323)

torch.manual_seed(3969)



image_transforms = transforms.Compose([

    transforms.Resize((100,100)),

    transforms.ToTensor()

])



train_datasets = torchvision.datasets.ImageFolder("../input/fruits-360_dataset/fruits-360/Training", transform = image_transforms)



train_loader = DataLoader(train_datasets,batch_size = 30, shuffle = True)



image_datasets = torchvision.datasets.ImageFolder("../input/fruits-360_dataset/fruits-360/Test", transform = image_transforms)



val_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [8922, 8923])



val_loader = DataLoader(val_dataset, batch_size = 30)

test_loader = DataLoader(test_dataset, batch_size = 30)
class BestModel(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.conv1 = nn.Conv2d(3, 90, 5)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(90, 80, 5)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(80, 70, 9)

        self.pool3 = nn.MaxPool2d(2, 2) #70*7

        

        self.linear1 = nn.Linear(7*7*70, 500)

        self.linear2 = nn.Linear(500, 200)

        self.linear3 = nn.Linear(200, 103)

        

        self.Dropout = nn.Dropout(0.5)

        

    def forward(self,x):

        x = func.relu(self.pool1(self.conv1(x)))

        x = func.relu(self.pool2(self.conv2(x)))

        x = func.relu(self.pool3(self.conv3(x)))

        

        x = x.view(-1, 7*7*70)

        L = 0.0001

        x = func.elu(self.linear1(x),L)

        x = self.Dropout(x)

        x = func.elu(self.linear2(x),L)

        x = self.Dropout(x)

        x = self.linear3(x)

        return x
model = BestModel()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()
def fit(model, train_loader, val_loader, optimizer, criterion, epochs, tries):

    device = dev("cuda:0")

    model = model.to(device)

    min_loss_v = 100

    

    for epoch in range(epochs):

        model.train()

        loss_v = 0

        epoch_loss = 0

        counter = 0

        for xx,yy in train_loader:

            xx = xx.cuda()

            yy = yy.cuda()

            optimizer.zero_grad()

            pred = model.forward(xx)

            loss = criterion(pred,yy)

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

        epoch_loss /= len(train_loader)

        with torch.no_grad():

            model.eval()

            for xx,yy in val_loader:

                xx = xx.cuda()

                yy = yy.cuda()

                pred = model.forward(xx)

                loss = criterion(pred,yy)

                loss_v += loss.item()

            loss_v /= len(val_loader)

            print("Epoch = ", epoch, ", Epoch_loss = ", epoch_loss, ", Val_loss = ", loss_v)

            if loss_v < min_loss_v:

                print("new min_loss_v")

                torch.save(model.state_dict(), "../best_model.md")

                min_loss_v = loss_v

            else:

                counter += 1

                print("counter = ", counter, "fail")

                if counter == tries:

                    print("GAME OVER")   

                    break

            

    state = torch.load("../best_model.md")  

    model.load_state_dict(state)    

    model.eval()

    model.cpu()
fit(model, train_loader, val_loader, optimizer, criterion, 15, 3)
model.eval()

preds = []

true = []

for xx,yy in val_loader:

    xx = xx.cuda()

    model.cuda()

    pred = model.forward(xx)

    preds.extend(pred.argmax(1).tolist())

    true.extend(yy.tolist())

print(classification_report(true, preds))