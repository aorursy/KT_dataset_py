import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
import scipy as sklearn

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(617)
torch.manual_seed(617)
if device =='cuda':
    torch.cuda.manual_seed_all(617)
trans = transforms.Compose([
    transforms.ToTensor(), #학습을 위해 dataset들을 텐서로 전환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.ImageFolder(root='../input/music-genres-classification/train', transform=trans)
test_data = torchvision.datasets.ImageFolder(root='../input/music-genres-classification/test', transform=trans)
batch_size = 100
data_loader = DataLoader(dataset = train_data, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers=2, # 데이터를 빨리 읽어오기 위한것
                         drop_last=True)
import matplotlib.pyplot as plt
# plt.imshow(data_loader.dataset[0][0].numpy().reshape((28,28,3)))
plt.imshow(data_loader.dataset[0][0].numpy().transpose(2,1,0)) ## 3,28,28 -> 28,28,3
plt.show() #normal을해서 원본사진과 색의 차이가 있슴
plt.imshow(torch.transpose(data_loader.dataset[0][0], 2, 0)) 
plt.show()
# print(data_loader.dataset[0][0].numpy())
test_set = DataLoader(dataset = test_data, 
                      batch_size = len(test_data),
                      shuffle=False)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True)
        )
        torch.nn.init.xavier_normal_(self.fc[0].weight)
        torch.nn.init.xavier_normal_(self.fc[2].weight)
        torch.nn.init.xavier_normal_(self.fc[4].weight)
        
    def forward(self, x):
        return self.fc(self.layer2(self.layer1(x)).view(-1, 64*7*7))
model = MyModel().to(device)
loss = torch.nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
total_batch = len(data_loader)
for epoch in range(80+1):
    model.train()
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        # one-hot encoding되어 있지 않음
        Y = Y.to(device)
      
        optimizer.zero_grad()
        hypothesis = model(X) #forward 계산
        cost = loss(hypothesis, Y) #cost
        cost.backward() #backpropagation
        optimizer.step() #갱신

        avg_cost += cost / total_batch #평균 error
    if epoch % 5==0:
        with torch.no_grad():
            model.eval()
            for _, data in enumerate(test_set):
                imgs, label = data
        #                 print(len(imgs), len(label))
                imgs = imgs.to(device)
                label = label.to(device)

                prediction = model(imgs)
                
                correct_prediction = torch.argmax(prediction, 1) == label
                accuracy = correct_prediction.float().mean()
                
                print('Epoch: {:03d} cost: {:.9f} acc: {:.9f}'.format(epoch, avg_cost, accuracy.item()))
            

print('Learning finished')
with torch.no_grad():
    model.eval()
    for _, data in enumerate(test_set):
        imgs, label = data
#                 print(len(imgs), len(label))
        imgs = imgs.to(device)
        label = label.to(device)

        prediction = model(imgs)

        correct_prediction = torch.argmax(prediction, 1) == label

        accuracy = correct_prediction.float().mean()
        print(accuracy.item())
submit=pd.read_csv('../input/music-genres-classification/submition_form.csv')
submit['label'] = torch.argmax(prediction, 1).cpu().numpy()
submit.to_csv('submission.csv',index=False, header=True)
submit['label'].value_counts()
submit
