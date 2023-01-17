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
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch

class MinstDataset(data.Dataset):
    def __init__(self):
        train = pd.read_csv("../input/train.csv")
        train_labels = train['label'].values
        train = train.drop("label",axis=1).values.reshape(42000,1,28,28)
        self.datalist = train
        self.labels = train_labels
    def __getitem__(self, index):
        return torch.Tensor(self.datalist[index].astype(float)), self.labels[index]
    def __len__(self):
        return self.datalist.shape[0]

train_Set = MinstDataset()
trainloader = torch.utils.data.DataLoader( dataset = train_Set , batch_size= 64 , shuffle = True)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
from torch.autograd import Variable
for epoch in range(20):
    for i, (images, labels) in enumerate(trainloader):
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' %(epoch+1, 10, i+1, loss.data[0]))

model.eval()
test = pd.read_csv("../input/test.csv")
test = test.values.reshape(28000,1,28,28).astype(float)
test = Variable(torch.Tensor(test))

pred = model(test)

_, predlabel = torch.max(pred.data, 1)
predlabel = predlabel.tolist()

predlabel = pd.DataFrame(predlabel)
predlabel.index = np.arange(28000) + 1
id = np.arange(28000) + 1
id = pd.DataFrame(id)
id.index = id.index + 1

predlabel = pd.concat([id,predlabel], axis=1)
predlabel.columns = ["ImageId", "Label"]

predlabel.to_csv('predict.csv', index= False)