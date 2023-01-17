import numpy as np

import torch

from torch.utils.data import DataLoader, Dataset

import torch.nn as nn

import torch.nn.functional as F

from torch.utils import data

from torchvision import models

import pandas as pd

from matplotlib import pyplot as plt

import csv



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_data = train_features.iloc[:,4:].to_numpy()

train_labels = train_targets_scored.iloc[:,1:].to_numpy()



test_data = test_features.iloc[:,4:].to_numpy()
class my_train_dataset():    



    def __init__(self, train_data, train_labels):



        super(my_train_dataset).__init__()

        

        

        self.X = np.pad(train_data,((0,0),(14, 14)), 'constant', constant_values=(0)).reshape(-1,1,30,30)       

        

        self.X = torch.from_numpy(self.X).float()

        self.Y = torch.from_numpy(train_labels).float()

        

            

    def __getitem__(self,index):

        

        image = self.X[index]

        label= self.Y[index]



        return image, label

        

    def __len__(self):

        return len(self.X)

    





class my_test_dataset():    



    def __init__(self, test_images):



        super(my_test_dataset).__init__()

        

        self.X = np.pad(test_data, ((0,0),(14, 14)), 'constant', constant_values=(0)).reshape(-1,1,30,30)       

        

        self.X = torch.from_numpy(self.X).float()

        

            

    def __getitem__(self,index):

        

        image = self.X[index]



        return image

        

    def __len__(self):

        return len(self.X)

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

      

        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)

        self.bn1 = nn.BatchNorm2d(num_features = 32, eps=1e-05, momentum=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)

        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 256, 5, padding = 2)

        self.bn3 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1)

        self.conv4 = nn.Conv2d(256, 512, 5, padding = 2)

        self.bn4 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1)

        self.conv5 = nn.Conv2d(512, 1024, 5, padding = 2)

        self.bn5 = nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1)



            

        self.hidden = nn.Linear(1024,512)

        self.output = nn.Linear(512, 206)

        

        

        

    def forward(self, x):



        x = self.bn1(F.relu(self.conv1(x)))

        x = self.bn2(self.pool(F.relu(self.conv2(x))))       

        x = self.bn3(F.relu(self.conv3(x)))

        x = self.bn4(F.relu(self.conv4(x)))

        x = self.bn5(F.relu(self.conv5(x)))



        x = F.avg_pool2d(x, [x.size(2), x.size(3)], stride=1)

       

        x = x.reshape(x.shape[0],x.shape[1])

       

        x = F.relu(self.hidden(x))

        x = self.output(x)

        

        return x
net = Net()

net.to(device)

net
Training_Loss = []



def train(model, data_loader, epochs):

    net.train()

    for epoch in range(epochs):

        

        for batch_num, (feats, labels) in enumerate(data_loader):

            feats, labels = feats.to(device), labels.to(device)

            

            outputs = model(feats)

            loss = criterion(outputs, labels)

            

            optimizer.zero_grad()

            

            loss.backward()

            optimizer.step()

            

            

            del feats

            del labels

            del loss

        

        lr_scheduler.step()            

            

        train_loss = test_classify(model, data_loader)

        print('Epoch:'+str(epoch)+' Train Loss: {:.4f}\t'.format(train_loss))

        Training_Loss.append(train_loss)

        

        

def test_classify(model, test_loader):

    model.eval()

    test_loss = []



    for batch_num, (feats, labels) in enumerate(test_loader):

        feats, labels = feats.to(device), labels.to(device)

        outputs = model(feats)



        loss = criterion(outputs, labels.float())

        

#         pred_labels = (outputs>0.5).long()

#         accuracy += torch.sum(torch.eq(pred_labels, labels)).item()

        

        test_loss.extend([loss.item()]*feats.size()[0])

        del feats

        del labels



    model.train()

    return np.mean(test_loss)

#Training Batch size

Batch_size = 256



# Loss Function

criterion = nn.BCEWithLogitsLoss()



# Optimizer

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)



# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 90, eta_min=0)  

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 25, gamma = 0.1)



# Epochs

num_Epochs = 90
#Train Dataloader

train_dataset = my_train_dataset(train_data,train_labels)          

train_dataloader = data.DataLoader(train_dataset, shuffle= True, batch_size = Batch_size, num_workers=4,pin_memory=True)





#Test Dataloader

test_dataset = my_test_dataset(test_data)

test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
train(net, train_dataloader, epochs = num_Epochs)
def predict(model, test_loader):

    

    model.eval()  

    prediction = []

    for batch_num, (feats) in enumerate(test_loader):

        feats = feats.to(device)

        outputs = model(feats)



        prediction.append(outputs.sigmoid().detach().cpu().numpy())

    prediction = np.concatenate(prediction)



    return prediction







prediction = predict(net, test_dataloader)

submission[submission.columns[1:]] = prediction

submission.to_csv("submission.csv", index = False)
plt.figure(figsize=(10,10))

x = np.arange(1,num_Epochs+1)

plt.plot(x, Training_Loss, label = 'Training Loss')

plt.xlabel('Epochs', fontsize =16)

plt.ylabel('Loss', fontsize =16)

plt.title('Loss v/s Epochs',fontsize =16)

plt.legend(fontsize=16)