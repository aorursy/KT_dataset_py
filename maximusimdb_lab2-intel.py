import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



path_to_train = "../input/seg_train/seg_train/"

path_to_test  = "../input/seg_test/seg_test/"



import torch

from torchvision import transforms

from torch.utils.data import TensorDataset,DataLoader

import torchvision

np.random.seed(12345)

torch.manual_seed(12345)

mean = [0.4,0.4,0.4]

std  = [0.2,0.2,0.2]

normalize = transforms.Compose([

        transforms.Resize(size=(160,160)),

        transforms.ToTensor(),

        transforms.Normalize(mean=mean, std=std, inplace=True)])



train_dataset =  torchvision.datasets.ImageFolder(path_to_train, transform=normalize)

train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)



val_dataset = torchvision.datasets.ImageFolder(path_to_test, transform=normalize)

val_ds, test_ds = torch.utils.data.random_split(val_dataset, [1500, 1500])

val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)



print(len(train_dataset), len(val_dataset), len(test_loader))
import torch.nn as nn

import torch.nn.functional as F

class network(nn.Module):

    def __init__(self):

        super(network, self).__init__()

        self.hidden = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=60, kernel_size=5), 

            nn.ReLU(), nn.MaxPool2d((4,4)),      

            nn.Conv2d(in_channels=60, out_channels=100, kernel_size=2), 

            nn.ReLU(), nn.MaxPool2d((11,12)))

        self.output = nn.Sequential(

            nn.Linear(900,100),

            nn.ReLU(),

            nn.Linear(100,6)

        )

    def forward(self, x):

        y = self.hidden(x).view(x.size(0), -1)

        return  self.output(y)
from sklearn.metrics import accuracy_score

def fit(net,crit,train_loader,val_loader,optimizer, epochs):

    best=0

    net.cuda()

    for i in range(epochs):

        tr_loss = 0

        val_loss = 0

        val_accuracy =0

        for xx,yy in train_loader:

            xx, yy = xx.cuda(), yy.cuda()

            optimizer.zero_grad()

            y = net.forward(xx)

            loss = crit(y,yy)

            tr_loss += loss

            loss.backward()

            optimizer.step()

        tr_loss /= len(train_loader)

        with torch.no_grad():

            for xx,yy in val_loader:

                xx, yy = xx.cuda(), yy.cuda()

                y = net.forward(xx)

                loss = crit(y,yy)

                val_loss += loss

                y_pred=y.argmax(dim=1).cpu().numpy()

                yy = yy.cpu().numpy()

                val_accuracy += accuracy_score(y_pred,yy)

            val_accuracy /= len(val_loader)

            if val_accuracy>best:

                best = val_accuracy

                torch.save(net.state_dict(), "../model.py")

        print(("epoch:%d, train:%f, val:%f" % (i,tr_loss.item(),val_accuracy.item())))

    net.cpu()

    print("Train ended. Best accuracy is %f" % float(best))
model = network()

from torch.optim import Adam

criterion = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.001)

fit(model,criterion,train_loader,val_loader,optimizer,10)
from sklearn import metrics

target_names = val_dataset.classes

all_preds = []

correct_preds = []

with torch.no_grad():

    model.eval()

    for xx, yy in test_loader:

        model.cuda()

        xx = xx.cuda()

        output = model.forward(xx)

        all_preds.extend(output.argmax(1).tolist())

        correct_preds.extend(yy.tolist())



print(metrics.classification_report(correct_preds, all_preds,target_names=target_names))


confusion_matrix = metrics.confusion_matrix(correct_preds, all_preds)

pd.DataFrame(confusion_matrix, index=target_names, columns=target_names)