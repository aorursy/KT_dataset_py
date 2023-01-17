import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

from torchvision import transforms

import os

path = "../input/fruits-360_dataset/fruits-360"

print(os.listdir(path))



np.random.seed(5315)

torch.manual_seed(9784)
channel_means = (0.485, 0.456, 0.406)

channel_stds = (0.229, 0.224, 0.225)

def to_tensor(a):

    res = transforms.Compose([

        transforms.Resize(size=(100,100)),

        transforms.ToTensor(),

        transforms.Normalize(mean=channel_means, std=channel_stds, inplace=True)])

    return res(a)



train_ds   =  torchvision.datasets.ImageFolder(path+"/Training", transform=to_tensor)

test_val_dataset = torchvision.datasets.ImageFolder(path+"/Test", transform=to_tensor)



val_size = int(0.7*len(test_val_dataset))

test_size = int(len(test_val_dataset) - val_size)

val_ds, test_ds = torch.utils.data.random_split(test_val_dataset, [val_size, test_size])
batch_size = 32

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)



print(len(train_ds),len(val_ds), len(test_ds))
from sklearn import metrics

class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.properties = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(3, padding=1),

            nn.BatchNorm2d(64),

            

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(3, padding=1),

            nn.BatchNorm2d(64),



            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            nn.MaxPool2d(4, padding=1),

            nn.BatchNorm2d(64))

        

        self.classifier = nn.Sequential(

            nn.Linear(576,1024),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(1024,512),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(512,103))

        

    def predict(self, inp):

        result = self.properties(inp).view(inp.size(0), -1)

        return self.classifier(result)

    def fit(self,optimizer,criterion,train_loader,val_loader,path_to_best_model,epoches,tolerance):

        self.cuda()

        best_accuracy = 0 #best accuracy on val_loader

        current_tolerance = tolerance

        train_len = len(train_loader)

        val_len = len(val_loader)

        for i in range(epoches):

            train_loss=0

            self.train()

            for xx,yy in train_loader:

                optimizer.zero_grad()

                xx,yy = xx.cuda(),yy.cuda()

                y_predicted = self.predict(xx)

                loss = criterion(y_predicted,yy)

                train_loss += loss.item()

                loss.backward()

                optimizer.step()

            train_loss /= train_len

            

            with torch.no_grad():

                self.eval()

                all_preds = []

                correct_preds = []

                for xx,yy in val_loader:

                    xx,yy = xx.cuda(),yy.cuda()

                    y_predicted = self.predict(xx)

                    all_preds.extend(y_predicted.argmax(1).tolist())

                    correct_preds.extend(yy.tolist())

                accuracy = metrics.accuracy_score(all_preds,correct_preds)

                if accuracy > best_accuracy:

                    best_accuracy = accuracy

                    current_tolerance = tolerance

                    torch.save(self.state_dict(),path_to_best_model)# "../network.py") 

                else:

                    current_tolerance -= 1

                if current_tolerance <= 0:

                    break

            print("Epoch:%d, train loss:%f, accuracy:%f, tolerance:%d" % (i, train_loss, accuracy, current_tolerance))

        self.eval()

        #torch.save(self.state_dict(),"../network1.py")

        self.load_state_dict(torch.load(path_to_best_model))

        print("Training stopped")
model = Network()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

criterion = nn.CrossEntropyLoss()

path_to_model = "../network.py"

model.fit(optimizer,criterion,train_loader,val_loader,path_to_model,epoches=15,tolerance=5)

model.eval()

all_preds = []

correct_preds = []

names = train_ds.classes

for xx,yy in test_loader:

    xx,yy = xx.cuda(),yy.cuda()

    y_predicted = model.predict(xx)

    all_preds.extend(y_predicted.argmax(1).tolist())

    correct_preds.extend(yy.tolist())

accuracy = metrics.accuracy_score(all_preds,correct_preds)

print(metrics.classification_report(all_preds,correct_preds,target_names=names))