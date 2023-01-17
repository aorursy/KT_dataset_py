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
%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



#import torchvision.transforms as transforms

from torch.utils.data import DataLoader,TensorDataset



import pandas as pd

import numpy as np



from sklearn.metrics import classification_report,confusion_matrix



import matplotlib.pyplot as plt
X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')
print('Shape of training dataset: ',X_train.shape)

print('Shape of testing dataset: ',X_test.shape)
X_train.describe()
X_test.describe()
y_train = X_train.pop('label')
X_train = X_train.values.reshape(-1,1,28,28)

X_test = X_test.values.reshape(-1,1,28,28)
print('Shape of training dataset: ',X_train.shape)

print('Shape of testing dataset: ',X_test.shape)
plt.bar(y_train.unique(),y_train.value_counts())

plt.title('Counts of each class (0-9)')

plt.xticks(y_train.unique(),y_train.unique())

plt.grid()
import torchvision.transforms as transforms
transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomAffine(15,(.1,.1)),

    transforms.ToTensor()

])
X_train = [transform(X.transpose(1,2,0).astype(np.uint8)).unsqueeze(0) for X in X_train]
X_train = torch.cat(X_train,0) * 255
y_train,X_test = map(torch.tensor,(y_train,X_test))
X_train = (X_train).float()

y_train = y_train.long()

X_test = (X_test).float()
train_dl = DataLoader(TensorDataset(X_train,y_train),

                                    batch_size=1024)
class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1,16,(3,3),padding=1)

        self.conv2 = nn.Conv2d(16,32,(3,3))

        self.conv3 = nn.Conv2d(32,64,(3,3),padding=1)

        self.conv4 = nn.Conv2d(64,128,(3,3))

        

        self.fc1 = nn.Linear(5*5*128,128)

        self.fc2 = nn.Linear(128,64)

        self.fc3 = nn.Linear(64,10)

    def forward(self,X):

        X = F.relu(self.conv1(X))

        X  = F.max_pool2d(F.leaky_relu(self.conv2(X)),(2,2))

        X = F.relu(self.conv3(X))

        X  = F.max_pool2d(F.leaky_relu(self.conv4(X)),(2,2))

        X = F.relu(self.fc1(X.view(-1,5*5*128)))

        X = F.relu(self.fc2(X))

        return self.fc3(X)

net = CNN()
lr = .005

optimizer = optim.Adam(net.parameters(),lr=lr)

loss_fn = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epochs = 20

for epoch in range(epochs):

    epoch_loss = 0

    scheduler.step()

    for xb,yb in train_dl:

        outcome = net(xb);

        loss = loss_fn(outcome,yb)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        with torch.no_grad():

            epoch_loss += len(xb) * loss

    print(epoch+1,"/",epochs," loss: ",epoch_loss/len(X_train))
with torch.no_grad():

    y_pred = net(X_test)
submission_df = pd.DataFrame()

submission_df['ImageId'] = np.arange(len(X_test))+1

submission_df['Label'] =  y_pred.argmax(dim=1)

submission_df.to_csv('submission.csv',index=False)