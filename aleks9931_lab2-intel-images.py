# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/seg_train/seg_train"))

print(os.listdir("../input/seg_test/seg_test"))



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



seed = 9931

np.random.seed(seed)

torch.manual_seed(seed)



data_path_format = '../input/seg_{0}/seg_{0}'



image_transforms = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.ToTensor()

])



train_datasets = torchvision.datasets.ImageFolder("../input/seg_train/seg_train/", transform = image_transforms)

train_loader = DataLoader(train_datasets,batch_size = 30, shuffle = True)



image_datasets = torchvision.datasets.ImageFolder("../input/seg_test/seg_test/", transform = image_transforms)



#val_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [6900, 6900])



devset_indices = np.arange(len(image_datasets))

devset_labels = image_datasets.targets



test_indices, val_indices, test_labels,  val_labels = model_selection.train_test_split(devset_indices, devset_labels, test_size=0.5, stratify=devset_labels, shuffle=True)



val_image_datasets = tdata.Subset(image_datasets, val_indices)

test_image_datasets = tdata.Subset(image_datasets, test_indices)



val_loader = DataLoader(val_image_datasets, batch_size = 30)

test_loader = DataLoader(test_image_datasets, batch_size = 30)



print(classification_report(test_labels,test_labels))

print(classification_report(val_labels,val_labels))
class BestModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 120, 5) # 3* 150 / 120* 146

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(120, 90, 4) # 120* 73 / 90* 70

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(90, 60, 4) # 90* 35 / 60* 32

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(60, 30, 3) # 60* 16 / 30* 14

        self.pool4 = nn.MaxPool2d(2, 2)

        # 30* 7

        self.linear1 = nn.Linear(30* 7*7, 300)

        self.linear2 = nn.Linear(300, 200)

        self.linear3 = nn.Linear(200, 100)

        self.linear4 = nn.Linear(100, 6)

        

        self.dropout = nn.Dropout(p = 0.4)

        

    def forward(self,x):

        x = self.pool1(self.conv1(x))

        x = func.relu(x)

        x = self.pool2(self.conv2(x))

        x = func.relu(x)

        x = self.pool3(self.conv3(x))

        x = func.relu(x)

        x = self.pool4(self.conv4(x))

        x = func.relu(x)

        

        x = x.view(-1, 30*7*7)

        

        x = func.relu(self.linear1(x))

        x = self.dropout(x)

        x = func.relu(self.linear2(x))

        x = self.dropout(x)

        x = func.relu(self.linear3(x))

        x = self.dropout(x)

        x = func.relu(self.linear4(x))

                

        return x
model = BestModel()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()
def train(model, train_loader, val_loader, optimizer, criterion, epochs, tries):

   

    device = dev("cuda:0")

    model = model.to(device)

    val_loss_best = 100

    check = 0

    

    for i in range(epochs):

        model.train()

        val_loss = 0

        epoch_loss = 0

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

                val_loss += loss.item()

            val_loss /= len(val_loader)

            print("Epoch = ", i, ", Epoch_loss = ", epoch_loss, ", Val_loss = ", val_loss)

            if val_loss < val_loss_best:

                print("Not bad, not bad! Keep going!")

                torch.save(model.state_dict(), "../best_model.md")

                val_loss_best = val_loss

            else:

                check += 1

                print("Very bad! Try again!")

                if check == tries:

                    print("You were my brother Model, I loved you!")   

                    break

            

    model.load_state_dict(torch.load("../best_model.md"))    

    model.eval()

    model.cpu()
train(model, train_loader, val_loader, optimizer, criterion, epochs = 30, tries = 10)
model.eval()

preds = []

true = []

for xx,yy in test_loader:

    xx = xx.cuda()

    model.cuda()

    pred = model.forward(xx)

    preds.extend(pred.argmax(1).tolist())

    true.extend(yy.tolist())

print(classification_report(true, preds))