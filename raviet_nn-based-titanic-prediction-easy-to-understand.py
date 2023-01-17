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
# Import train & test Data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
dataset=[train,test]
for data in dataset:

    data['lastname']=data['Name'].apply(lambda x: x.split(',')[0])
count=0

lastnamerecorder=[]

for i in range(train.shape[0]):

    if( train['lastname'][i] not in lastnamerecorder):

        count=count+1+train['SibSp'][i]+train['Parch'][i]

        lastnamerecorder.append(train['lastname'][i])

print(count)
lastname_cabin={}

for i in range(train.shape[0]):

    if( train['lastname'][i] not in lastname_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):

        lastname_cabin[train['lastname'][i]]=[train['Cabin'][i]]

    elif(train['lastname'][i] in lastname_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):

        if(train['Cabin'][i] not in lastname_cabin[train['lastname'][i]] ):

            lastname_cabin[train['lastname'][i]].append(train['Cabin'][i])

lastname_cabin
class_cabin={}

for i in range(train.shape[0]):

    if( train['Pclass'][i] not in class_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):

        class_cabin[train['Pclass'][i]]=[train['Cabin'][i]]

    elif(train['Pclass'][i] in class_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):

        if(train['Cabin'][i] not in class_cabin[train['Pclass'][i]] ):

            class_cabin[train['Pclass'][i]].append(train['Cabin'][i])

class_cabin
for data in dataset:

    data["cabin"]=data['Cabin'].apply(lambda x: x[0] if x==x else x)

data["cabin"]
d=['X','A','B','C','D','E','F','G']
for data in dataset:

    for i in range(data.shape[0]):

        if(data['Pclass'][i]==1 and data["cabin"][i]!=data["cabin"][i]):

            data['cabin'][i]=d[np.random.choice(np.arange(1, 6), p=[0.3,0.3,0.2,0.15,0.05])]

        elif(data['Pclass'][i]==2 and data["cabin"][i]!=data["cabin"][i]):

            data['cabin'][i]=d[np.random.choice(np.arange(4, 7), p=[0.4,0.3,0.3])]

        elif(data['Pclass'][i]==3 and data["cabin"][i]!=data["cabin"][i]):

            data['cabin'][i]=d[np.random.choice(np.arange(5, 8), p=[0.3,0.4,0.3])]
for data in dataset:

    data['Age']=data['Age'].fillna(data['Age'].mean())
test=test.drop(["Cabin"], axis=1)

train=train.drop(["Cabin"], axis=1)
for data in dataset:

    data['family']=data['Parch']+data['SibSp']
columns=['PassengerId','Name','Ticket','lastname']

train=train.drop(columns,axis=1)

test1=test.drop(columns, axis=1)
train=pd.get_dummies(train)

test1=pd.get_dummies(test1)
train
X=train.drop(['Survived'],axis=1)



Y=train['Survived']
X.info()
import torch

import numpy as np
inputs=X.to_numpy()

targets=Y.to_numpy()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

inputs=scaler.fit_transform(inputs)

inputs
inputs = torch.from_numpy(inputs)

targets = torch.from_numpy(targets)
inputs=inputs.float()

##targets=targets.float()
inputs.size()
targets.size()
import torch.nn as nn

from torch.utils.data import TensorDataset
train_ds = TensorDataset(inputs, targets)
from torch.utils.data import DataLoader
batch_size = 100

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.linear1 = nn.Linear(18, 32)

        

        self.linear2 = nn.Linear(32, 32)

        self.linear3 = nn.Linear(32, 2)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        x = self.linear1(x)

        x = self.linear2(x)

        x= self.linear3(x)

        x = self.sigmoid(x)

        return x
model=Net()

preds=model.forward(inputs)
preds
import torch.nn.functional as F

loss_fn = F.cross_entropy

loss = loss_fn(model.forward(inputs), targets)

opt = torch.optim.SGD(model.parameters(), lr=1e-2)
def fit(num_epochs, model, loss_fn, opt, train_dl):

    

    # Repeat for given number of epochs

    for epoch in range(num_epochs):

        

        # Train with batches of data

        for xb,yb in train_dl:

            

            # 1. Generate predictions

            pred = model.forward(xb.float())

            

            # 2. Calculate loss

            loss = loss_fn(pred, yb)

            

            # 3. Compute gradients

            loss.backward()

            

            # 4. Update parameters using gradients

            opt.step()

            

            # 5. Reset the gradients to zero

            opt.zero_grad()

        

        # Print the progress

        if (epoch+1) % 10 == 0:

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
fit(1000, model, loss_fn, opt,train_dl)
preds=model.forward(inputs)

preds
targets
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
accuracy(preds,targets)
test1['cabin_T']=0
testf=scaler.fit_transform(test1.to_numpy())
testf = torch.from_numpy(testf)
final=model.forward(testf.float())
_, preds = torch.max(final, dim=1)

preds.size()
testf
final=preds.numpy()
final
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final})

output.to_csv('Titanic_submission.csv', index=False)