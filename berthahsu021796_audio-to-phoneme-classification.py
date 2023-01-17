from google.colab import files
!pip install -q kaggle
uploaded = files.upload()
!ls
!pwd
%cd ..
!ls
!mkdir root/.kaggle/
!cp content/kaggle.json root/.kaggle/kaggle.json
!kaggle competitions download -c 11-785-s20-hw1p2
!ls
!unzip \*.zip
import numpy as np
import pandas as pd

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt
import time

cuda = torch.cuda.is_available()
cuda
#loading dataset into numpy object
train_data = np.load('train.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy', allow_pickle = True)
val_data = np.load('dev.npy', allow_pickle=True)
val_labels = np.load('dev_labels.npy', allow_pickle=True)
test_data = np.load('test.npy', allow_pickle=True)
#data summary
print(train_data.shape, train_labels.shape)
print(val_data.shape, val_labels.shape)
print(test_data.shape)
class MyDataset(data.Dataset):

    def __init__(self, X, Y, k):

        # X: data numpy object
        # Y: label numpy object
        # k: number of frames to be padded both sides of each frame
        self.utter_len = dict()
        self.X = dict()
        self.Y = dict()
        self.k = k
        self.length = 0
        pad = np.zeros((k,40))

        #for each utterance
        for i in range(len(X)):

          #storing (start_idx, end_idx) of each utterance in utter_len dict
          old_len = self.length
          self.length += X[i].shape[0]
          self.utter_len[i] = (old_len, self.length-1)

          self.X[i] = torch.from_numpy(np.concatenate((pad, X[i], pad), axis = 0)).float()
          self.Y[i] = torch.from_numpy(Y[i]).long()

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        #binary search for the right pair given an index      
        left = 0
        right = len(self.X)
        while right-left > 0:
          mid = int((left+right)/2)
          #if index in the (mid)th row: return the vector
          if self.utter_len[mid][0] <= index and self.utter_len[mid][1] >= index:
            idx = index-self.utter_len[mid][0]
            x = self.X[mid][idx:idx+2*self.k+1].reshape(-1)
            y = self.Y[mid][idx]
            return (x,y)
          else:
            if self.utter_len[mid][0] > index:
              right = mid
            if self.utter_len[mid][1] < index:
              left = mid+1
      
#create dataloaders

#adding context
k = 12
num_workers = 8 if cuda else 0 
    
# Training
train_dataset = MyDataset(train_data, train_labels, k)

train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=64)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Validating
val_dataset = MyDataset(val_data, val_labels, k)

val_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
val_loader = data.DataLoader(val_dataset, **val_loader_args)

#delete to obtain more memory
del train_data
del train_labels
del val_data
del val_labels
class MyNetwork(torch.nn.Module):

    #optimizing techniques
    #increase n of layers 4 or 5
    #decrease hidden neurons
    #reduce lr on plateau torch
    #adding drop out
    #ensemble

    def __init__(self, size_list):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size_list[i+1]))
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, input_val):
        return self.net(input_val)
#creating model
n = (2*k+1)*40
model = MyNetwork([n, 2048, 1024, 1000, 512, 256, 138])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 2, factor = 0.1)
device = torch.device("cuda" if cuda else "cpu")
print(device)
model.to(device)
print(model)
def train_epoch(model, train_loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data)
        
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss
def test_model(model, val_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(val_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc
#start training for 30 epochs
n_epochs = 30
Train_loss = []
Val_loss = []
Val_acc = []

for i in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = test_model(model, val_loader, criterion)
    Train_loss.append(train_loss)
    Val_loss.append(val_loss)
    Val_acc.append(val_acc)
    scheduler.step(val_loss)
    print('='*20)
    if val_acc > 63.0:
      break
class TestDataset(data.Dataset):

    def __init__(self, X, k):

        self.utter_len = dict()
        self.X = dict()
        self.k = k
        self.length = 0
        pad = np.zeros((k,40))

        #for each utterance
        for i in range(len(X)):

          #storing (start_idx, end_idx) of each utterance in utter_len dict
          old_len = self.length
          self.length += X[i].shape[0]
          self.utter_len[i] = (old_len, self.length-1)

          self.X[i] = torch.from_numpy(np.concatenate((pad, X[i], pad), axis = 0)).float()

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        #binary search for the right pair given an index      
        left = 0
        right = len(self.X)
        while right-left > 0:
          mid = int((left+right)/2)
          #if index in the (mid)th row: return the vector
          if self.utter_len[mid][0] <= index and self.utter_len[mid][1] >= index:
            idx = index-self.utter_len[mid][0]
            x = self.X[mid][idx:idx+2*self.k+1].reshape(-1)
            return x
          else:
            if self.utter_len[mid][0] > index:
              right = mid
            if self.utter_len[mid][1] < index:
              left = mid+1
      
test_dataset = TestDataset(test_data, k)

test_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
                    
test_loader = data.DataLoader(test_dataset, **test_loader_args)
def predict_model(model, test_loader):

    results = []
    with torch.no_grad():
        model.eval()

        for batch_idx, data in enumerate(test_loader):   
            data = data.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            results.append(predicted)
      
    ans = pd.DataFrame(columns=["id", "label"])
    idx = 0
    for i in range(len(results)):
      for j, label in enumerate(results[i]):
        ans = ans.append({'id': idx, 'label':label.item()}, ignore_index=True)
        idx += 1

    ans_csv = ans.to_csv('result.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
    print(ans)
#use trained model to predict on test data
predict_model(model, test_loader)
files.download('result.csv') 