# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import torch.nn.functional as F
import torch
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
train_data = train_data.drop(["Ticket"], axis=1)

Y = train_data["Survived"]
print(Y[0:10])
print(Y.values)
train_data.head()
def extract_title(table):
    title_dic = {
        "Capt": "Army",
        "Col" : "Army",
        "Major": "Army",
        "Jonkheer":"Noble",
        "Don":"Noble",
        "Sir":"Noble",
        "the Countess":"Noble",
        "the Count":"Noble",
        "Donna":"Noble",
        "Lady":"Noble",
        "Dr": "Prof",
        "Rev": "Prof",
        "Master": "Prof",
        "Mme": "People",
        "Mlle": "People",
        "Ms": "People",
        "Mr" : "People",
        "Mrs" : "People",
        "Miss" : "People"
    }
    table['Title'] = table['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

extract_title(train_data)
extract_title(test_data)

test_data = test_data.drop(["Name","Ticket"], axis=1)
train_data = train_data.drop(["Name"], axis = 1)
train_data.head()
def series_convert(s):
    n = s.fillna(0)
    s = n.str.slice(start=0,stop=1)
    return s    
train_data["Cabin"] = series_convert(train_data["Cabin"])
def age_fill(data):
    mean_age_by_title = data.groupby('Title').mean()['Age']
    for title, age in mean_age_by_title.iteritems():
        if age>18:
            data.loc[data['Age'].isnull() & (data['Title']==title), 'Age']=age
age_fill(train_data)
age_fill(test_data)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","Cabin", "Title"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
print(X[0:10])
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

tensorX = torch.tensor(X.values)
tensorY = torch.tensor(Y.values)
tensorY = torch.flatten(tensorY)
tensorY = tensorY.unsqueeze(1).float()
# TensorY.float()
# tensorY = torch.view(tensorY, -1)
# print(tensorY.size)
# print(tensorX.size)
# # tensorY = torch.reshape(tensorY,(-1,))
# print(tensorY)
# print(tensorX)
# X_train, X_valid = train_set, val_set = torch.utils.data.random_split(tensorX, [50000, 10000])

X_train, X_valid, Y_train, Y_valid = train_test_split(tensorX,
                                                    tensorY,
                                                    test_size=0.25,
                                                    random_state=42)


data_train = TensorDataset(X_train, Y_train)
data_valid = TensorDataset(X_valid, Y_valid)
batch_size = 16
#     #Crate a dataloader -- Note Shuffle has been set to True but needs to set False if used for Validation/Test
# data_load_train = DataLoader(data_train, shuffle=True, batch_size=batch_size)
# data_load_valid = DataLoader(data_valid, shuffle=True, batch_size=batch_size)

# data = TensorDataset(tensorX, tensorY)
dataload = {'train': DataLoader(data_train, shuffle=True, batch_size=batch_size), 
             'valid': DataLoader(data_valid, shuffle=False, batch_size=batch_size) }
train_on_gpu = torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        ## Define layers of a CNN
        self.fc1a = nn.Linear(35,64)
        self.batch_fc1a = nn.BatchNorm1d(64)
        self.fc2a= nn.Linear(64,128)
        self.batch_fc2a = nn.BatchNorm1d(128)
#         self.conv1 = nn.Conv1d(8,16,1, padding=1)
#         self.batch_conv1 = nn.BatchNorm1d(16)
#         self.conv2 = nn.Conv1d(16,32,1,padding=1)
#         self.batch_conv2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(128,256)
        self.batch_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,128)
        self.batch_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,1)
        self.dropout = nn.Dropout(0.45)
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.batch_fc1a(self.fc1a(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_fc2a(self.fc2a(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x    
model = Net()

if use_cuda:
    model.cuda()

print(model)
import torch
import torch.optim as optim

### TODO: select loss function
criterion = nn.BCEWithLogitsLoss()

### TODO: select optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            optimizer.zero_grad()
            
            output = model(data.float())
            
            loss = criterion(output, target)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        model.eval()
        for batch_idex, (data, target) in enumerate(loaders["valid"]):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            output = model(data.float())
            
            loss = criterion(output, target)
            
            valid_loss = valid_loss +((1/(batch_idx+1))*(loss.data - valid_loss))
            
            
            
            
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss, valid_loss
            ))
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss 
            
    return model
            
model_test = train(5, dataload, model, optimizer, criterion, use_cuda, 'model_test.pt')
