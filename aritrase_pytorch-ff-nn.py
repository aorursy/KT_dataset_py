# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader



import lightgbm as lgb

from sklearn.metrics import mean_squared_error



import os

print(os.listdir("../input"))
#Load data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.shape, test_df.shape
type(train_df)
X_train = train_df.drop(['target','ID_code'], axis = 1)

x_test = test_df.drop(['ID_code'],axis = 1)

Y_train = train_df['target']
#### Scaling feature #####

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

x_test = sc.transform(x_test)
X_train.shape , Y_train.shape , x_test.shape
type(X_train)
batch_size = 256



x_train = torch.tensor(X_train , dtype=torch.float).cuda()

y_train = torch.tensor(Y_train, dtype=torch.long).cuda()

x_test = torch.tensor(x_test , dtype=torch.float).cuda()





train = torch.utils.data.TensorDataset(x_train, y_train)

#valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

#valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
# Let's check the shape of the input/target data

for data, target in train_loader:

    print(data.shape)

    print(target.shape)

    print(target.dtype)

    break
class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()

        self.fc1 = nn.Linear(200,100)

        self.fc2 = nn.Linear(100,50)

        self.fc3 = nn.Linear(50,30)

        self.fc4 = nn.Linear(30,2)

        

        self.dropout = nn.Dropout(p=0.3)

        

    def forward(self,x):

        # input tensor is flattened 

        x = x.view(x.shape[0], -1)

        

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = self.fc4(x)

        

        return x  
model = Model()

model.cuda()

criterion = nn.CrossEntropyLoss()



from torch import optim



optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1, 25): ## run the model for 15 epochs

    train_loss, valid_loss = [], []

    ## training part 

    model.train()

    for data, target in train_loader:

        optimizer.zero_grad()

        ## 1. forward propagation

        output = model(data)

        

        ## 2. loss calculation

        loss = criterion(output, target)

        

        ## 3. backward propagation

        loss.backward()

        

        ## 4. weight optimization

        optimizer.step()

        

        train_loss.append(loss.item())

        

    ## evaluation part

    #with torch.no_grad():

    #    model.eval()

    #    for data, target in valid_loader:

    #        output = model(data)

    #        loss = criterion(output, target)

    #        valid_loss.append(loss.item())

    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss))
out = model(x_test)

prediction = torch.max(out, 1)[1]

cpu_pred = prediction.cpu()

pred_y = cpu_pred.data.numpy()
id_code_test = test_df['ID_code']
my_submission_nn = pd.DataFrame({"ID_code" : id_code_test, "target" : pred_y})

my_submission_nn.to_csv('submission_nn.csv', index = False, header = True)