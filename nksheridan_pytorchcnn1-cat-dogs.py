# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import cv2
from tqdm import tqdm
import torch
REBUILD_DATA = True
#data processing class
class DogsVSCats():
    IMG_SIZE = 50
    CATS = "/kaggle/input/dogs-and-cats-fastai/dogscats/train/cats"
    DOGS = "/kaggle/input/dogs-and-cats-fastai/dogscats/train/dogs"
    TESTING = "/kaggle/input/dogs-and-cats-fastai/dogscats/test1"
    LABELS = {CATS: 0, DOGS: 1} 
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    #print(np.eye(2)[self.LABELS[label]])
                    
                    if label == self.CATS:
                        self.catcount +=1
                    elif label == self.DOGS:
                        self.dogcount +=1
            
            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
            print('Cats:', dogsvcats.catcount)
            print('Dogs:', dogsvcats.dogcount)
                   # except Exception as e:
                      #  pass
                        
dogsvcats = DogsVSCats()
dogsvcats.make_training_data()
#loading training data
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))
#Splitting training data in X and y, convert to tensor
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
import matplotlib.pyplot as plt

plt.imshow(X[0])
print(y[0])
import torch
import torch.nn as nn
import torch.nn.functional as F

# Now to build CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512,2)
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
net = Net()
print(net)
#now for training the CNN
import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
                  
VAL_PCT = 0.5
val_size = int(len(X)*VAL_PCT)
print(val_size)
print(len(X))
print(len(y))
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X), len(test_X))
BATCH_SIZE = 100
#try various epochs to see if validation accuracy increases
EPOCHS = 20

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_Y = train_y[i:i+BATCH_SIZE]
        
        net.zero_grad()
        
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch: {epoch}. Loss: {loss}") 
#validation for the CNN
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))
print(len(test_X))