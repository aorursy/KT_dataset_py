import numpy as np

import pandas as pd

import os



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))

import torch

import torchvision

from torchvision import transforms

from torch.utils.data import TensorDataset,DataLoader

def to_tensor(pil):

    res = transforms.Compose([

        transforms.Resize(size=(150,150)),

        transforms.ToTensor()])

    return res(pil)



train_ds = torchvision.datasets.ImageFolder("../input/seg_train/seg_train/", transform=to_tensor)

train_loader = DataLoader(train_ds,batch_size=32, shuffle=True)
val_dataset = torchvision.datasets.ImageFolder("../input/seg_test/seg_test/", transform=to_tensor)

val_ds, test_ds = torch.utils.data.random_split(val_dataset, [1500, 1500])



val_loader = DataLoader(val_ds, batch_size=32)

test_loader = DataLoader(test_ds, batch_size=32)
import torch.nn.functional as F

import torch.nn as nn

class intel(nn.Module):

    def __init__(self):

        super(intel, self).__init__()

        self.conv1 = nn.Conv2d(3, 30, 5)

        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(30, 50, 4)

        self.pool2 = nn.MaxPool2d(5, 2)

        self.conv3 = nn.Conv2d(50, 70, 3)

        self.pool3 = nn.MaxPool2d(3, 3)

        self.conv4 = nn.Conv2d(70, 75, 3)

        self.pool4 = nn.MaxPool2d(3, 4)

        

        self.classifier1 = nn.Linear(300, 300)

        self.classifier2 = nn.Linear(300, 6)

        

    def forward(self,x):

        a = self.conv1(x)

        a = self.pool1(a)

        a = a.relu()

        

        b = self.conv2(a)

        b = self.pool2(b)

        b = b.relu()

        

        c = self.conv3(b)

        c = self.pool3(c)

        c = c.relu()

        

        d = self.conv4(c)

        d = self.pool4(d)

        d = d.relu()

        

        e = d.view(x.size(0), -1)

        e = self.classifier1(e)

        e = e.relu()

        

        f = self.classifier2(e)

        out = F.log_softmax(f,dim=1)

        return out
model = intel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

print(device)
def train(model,train_ds,val_ds,optimizer, epochs, tolerance):

    running_tolerance = tolerance

    val_loss_best = 555

    criterion = nn.CrossEntropyLoss()

    for i in range(epochs):

        model.train()

        epoch_loss = 0

        val_loss = 0

        for xx,yy in train_ds:

            xx, yy = xx.cuda(), yy.cuda()

            batchsize = xx.size(0)

            optimizer.zero_grad()

            y = model.forward(xx)

            loss = criterion(y,yy)

            epoch_loss += loss

            loss.backward()

            optimizer.step()

        epoch_loss /= len(train_loader)

        with torch.no_grad():

            model.eval()

            for xx,yy in val_ds:

                xx, yy = xx.cuda(), yy.cuda()

                batchsize = xx.size(0)

                y = model.forward(xx)

                loss = criterion(y,yy)

                val_loss += loss

            val_loss /= len(val_loader)

            status = "epoch=%d, loss=%f, val_loss=%f, best_loss=%f" % (i,epoch_loss,val_loss,val_loss_best)

            print(status)

            if val_loss<val_loss_best:

                torch.save(model.state_dict(), "../best_model.md")

                val_loss_best = val_loss

                running_tolerance = tolerance

            else:

                running_tolerance -=1

                if running_tolerance==0:

                    print("Stop training")   

                    break

                print("Running tolerance is ", str(running_tolerance), "best is ",str(val_loss_best))

            

    model.load_state_dict(torch.load("../best_model.md"))    

    model.eval()

    model.cpu()
train(model,train_loader,val_loader,optimizer,25,tolerance=7)
from sklearn.metrics import classification_report



model.eval()

all_preds = []

correct_preds = []

for xx,yy in test_loader:

    xx = xx.cuda()

    model.cuda()

    y_pred = model.forward(xx)

    all_preds.extend(y_pred.argmax(1).tolist())

    correct_preds.extend(yy.tolist())

print(classification_report(all_preds,correct_preds))