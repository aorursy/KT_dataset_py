import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms 
from torch.autograd import Variable
import torch.nn.functional as F
num_epochs = 5
batch_size = 10
learning_rate = 0.001

class CustomedDataSet(torch.utils.data.Dataset):
    def getTrainY(self):
        return self.trainY
    def __init__(self, train=True):
        self.train = train
        if self.train :
            dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
            
            trainY = dataset.iloc[:,0].values
            trainX = dataset.iloc[:,1:].values.reshape(42000 ,1,28,28)
        
            trainX = trainX[:-1000]
            trainY = trainY[:-1000]
            self.datalist = trainX
            self.labellist = trainY
        else:
            dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
            
            trainY = dataset.iloc[:,0].values
            trainX = dataset.iloc[:,1:].values.reshape(42000 ,1,28,28)
            
            valX = trainX[-1000:]
            valY = trainY[-1000:]
            self.datalist = valX
            self.labellist = valY
            
    def __getitem__(self, index):    
        return torch.Tensor(self.datalist[index].astype(float)),self.labellist[index]
    
    def __len__(self):
        return self.datalist.shape[0]

train_dataset = CustomedDataSet()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

test_dataset = CustomedDataSet(train=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,num_workers=2)

dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
dataset.shape
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1 ,16, kernel_size=5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16 ,32, kernel_size=5,padding=2)
    
        self.fc1 = nn.Linear( 32 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 84)
        self.fc4 = nn.Linear(84, 10)
#         self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.norm(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#         print(x.shape)
        x = x.view(-1, 32 * 7 * 7)# 32, 7, 7
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x


cnn = Net()#.cuda()
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(cnn.parameters(),lr=0.001)
optimizer = torch.optim.SGD(cnn.parameters(),lr=learning_rate,momentum=0.9)

for epoch in range(num_epochs):
    running_loss=0
    for i, (images, labels) in enumerate(train_loader):
        g = images
        optimizer.zero_grad()
        images = Variable(images)#.cuda()
        labels = Variable(labels)#.cuda()
        
        output = cnn(images)
        
        loss = criterion(output,labels.long())
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if (i+1) % 1000 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, running_loss/1000))
            running_loss=0
            val_loss=0
            _ = cnn.eval()
            for i, (images, labels) in enumerate(test_loader):
                g = images
                optimizer.zero_grad()
                images = Variable(images)#.cuda()
                labels = Variable(labels)#.cuda()
                
                output = cnn(images)

                loss = criterion(output,labels.long())
                val_loss+=loss.item()
            print("Validataion Loss [%0.4f]"%(val_loss/i))
            _ = cnn.train()
import matplotlib.pyplot as plt
_ = cnn.eval()
for i, (images, labels) in enumerate(test_loader):
    g = images
    plt.imshow((g[0][0]).cpu().detach().numpy())
    output = cnn(images)
    plt.xlabel("Predicted:- "+str(torch.argmax(output[0]).detach().numpy())+" Ground-Truth:- "+str(labels[0].detach().numpy()))
    plt.show()
    if(i>20):
        break
!ls /kaggle/input