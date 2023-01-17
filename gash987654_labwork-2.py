import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision

import os

import torch

import torchvision

from torchvision import transforms

from torch.utils.data import TensorDataset,DataLoader

import torch.nn as nn

import torch.nn.functional as F



print(os.listdir("../input"))

print(os.listdir("../input/seg_train/seg_train/"))



channel_means = (0.485, 0.456, 0.406)

channel_stds = (0.229, 0.224, 0.225)

torch.manual_seed(123)

np.random.seed(123)
f = transforms.Compose([

        transforms.Resize(size=(140,140)),

        transforms.ToTensor(),

        transforms.Normalize(channel_means,channel_stds )])



batch_size = 16

train_dataset = torchvision.datasets.ImageFolder("../input/seg_train/seg_train/", transform=f,target_transform=None)

test = torchvision.datasets.ImageFolder("../input/seg_test/seg_test/", transform=f,target_transform=None)

test_len = int(len(test)/2)

val_len = int(len(test) - test_len)

val_dataset, test_dataset = torch.utils.data.random_split(test, [val_len, test_len])



val_loader =   DataLoader(val_dataset,  batch_size=batch_size, shuffle=True)

test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)



a = len(train_dataset)

b = len(val_dataset)

c = len(test_dataset)

print("Train len   ",a)

print("Val   len   ",b)

print("Test  len   ",c)
from sklearn.metrics import accuracy_score,classification_report

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.input = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=(4,4)), nn.ReLU(),

            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3,3)),nn.ReLU(),

            nn.AvgPool2d(2),

            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3,3)),nn.ReLU(),

            nn.MaxPool2d(10))

        self.output = nn.Sequential(

            nn.Linear(50,256), nn.ReLU(),

            nn.Linear(256,6)

        )

    def forward(self, x):

        result = self.input(x).view(x.size(0), -1)

        return self.output(result)

        
def train_model(model,optimizer, criterion, train_dl,val_dl,epochs):

    best_accuracy=0

    model.cuda()

    for epoch in range(epochs):

        model.train()

        running_train_loss = 0.0

        

        for xx, yy in train_dl:

            optimizer.zero_grad()

            xx, yy = xx.cuda(), yy.cuda()

            out = model.forward(xx)

            loss = criterion(out, yy)

            running_train_loss += loss.item()

            loss.backward()

            optimizer.step()

            

        with torch.no_grad():

            model.eval()

            running_corrects = 0

            running_total = 0

            running_loss = 0.0

            for xx, yy in val_dl:

                batch_size = xx.size(0)

                xx, yy = xx.cuda(), yy.cuda()

                out = model.forward(xx)

                loss = criterion(out, yy)

                running_loss += loss.item()

                predictions = out.argmax(1)

                running_corrects += (predictions == yy).sum().item()

                running_total += batch_size

            

            mean_val_loss = running_loss / len(val_dl)

            accuracy = running_corrects / running_total

            

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                torch.save(model.state_dict(), "../classidier.py")

            

            print(f"Epoch {epoch}, val_loss {mean_val_loss}, accuracy = {accuracy}")
classifier = Model()

criterion = nn.CrossEntropyLoss()

from torch.optim import Adam

optimizer = Adam(classifier.parameters(), lr=0.001)

train_model(classifier,optimizer,criterion,train_loader,val_loader,5)
classifier.cuda()

all_preds = []

correct_preds = []

loss=0

for xx,yy in test_loader:

    xx = xx.cuda()

    yy = yy.cuda()

    res = classifier.forward(xx)

    all_preds.extend(res.argmax(1).tolist())

    correct_preds.extend(yy.tolist())

classifier.cpu()

print("Test accuracy = ",accuracy_score(all_preds,correct_preds))

print((classification_report(all_preds,correct_preds)))
