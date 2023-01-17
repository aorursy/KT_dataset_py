import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import RobustScaler
data=pd.read_csv("../input/dont-overfit-ii/train.csv")

test = pd.read_csv("../input/dont-overfit-ii/test.csv")
rs = RobustScaler()
df=data.iloc[:,1:]
df.head()
from torch.utils.data import DataLoader,Dataset

import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F
class Csn(Dataset):

    def __init__(self,train_df):

        self.df=train_df

        

    def __len__(self):

        return(len(self.df))

    

    def __getitem__(self,idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        data  = self.df.iloc[idx,1:].values

        data = data[11:].reshape(17,17)

        label = self.df.iloc[idx].target

        

        return data,label

        
data = Csn(df)

dataloader = DataLoader(data,

                        shuffle=True,

                        num_workers=0,

                        batch_size=8)
image, label = next(iter(dataloader))

print(label[0])

plt.imshow(image[0,:])
class cnn(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1= nn.Conv2d(1,8,5,padding=2)

        self.conv2 = nn.Conv2d(8,16,(3,3),padding = 1)

        self.fc1 = nn.Linear(16*17*17 ,500)

        self.fc2 = nn.Linear(500,1)

        self.dropout = nn.Dropout(0.25)

        nn.init.kaiming_normal_(self.conv1.weight)

        nn.init.kaiming_normal_(self.conv2.weight)

        nn.init.kaiming_normal_(self.fc1.weight)

        nn.init.kaiming_normal_(self.fc2.weight)

        nn.init.zeros_(self.conv1.bias)

        nn.init.zeros_(self.conv2.bias)

        nn.init.zeros_(self.fc1.bias)

        nn.init.zeros_(self.fc2.bias)

        

    def forward(self,x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = x.view(-1, 16*17*17)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        

        return x

        

        

    
model1 = cnn()

model1
criterion=nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model1.parameters(), lr=0.003)

epochs= 100
model1=model1.float()
import warnings

warnings.filterwarnings('ignore')
x=0

for e in range(epochs):

    run_loss=0

    for image,labels in dataloader:

        optimizer.zero_grad()

        image = image.unsqueeze(1)

        output = model1(image.float())

        labels = labels.unsqueeze(1)

        loss = criterion(output, labels.float())

        loss.backward()

        optimizer.step()

        

        run_loss += loss.item()

    else:

        if x%10==0:

            print(f"Training loss for epoch {x}: {run_loss/len(dataloader)}")

        x=x+1;
test.head()
class Csn_test(Dataset):

    def __init__(self,train_df):

        self.df=train_df

        

    def __len__(self):

        return(len(self.df))

    

    def __getitem__(self,idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        data  = self.df.iloc[idx,1:].values

        data = data[11:].reshape(17,17)

        

        return data
data = Csn_test(test)

testloader = DataLoader(data,

                        shuffle=False,

                        num_workers=0,

                        batch_size=1)
image = next(iter(testloader))

plt.imshow(image[0,:])
lis=[]

model1.eval()

with torch.no_grad():

    for image in testloader:

        image = image.unsqueeze(1)

        output = model1(image.float())

        output = torch.sigmoid(output)

        lis.append(output.numpy())
y_pred_list=lis

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
sub=pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')

sub['target'] = y_pred_list

sub.to_csv('submit1.csv', index = False)