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
import collections
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

submission = pd.read_csv("../input/titanic/gender_submission.csv")
train = pd.read_csv("../input/titanic/train.csv")
train_len = len(train)

test = pd.read_csv("../input/titanic/test.csv")
test_len = len(test)


data = pd.concat([train,test], sort = False)
Sex = collections.Counter(data["Sex"])
Embarked = collections.Counter(data["Embarked"])
Cabin = collections.Counter(data["Cabin"])
print(Embarked)
print(Cabin)

#前処理は欠損値の処理をしていく
#Cabin列は削除
#Embarkedの欠損値列はSで埋める
data["Sex"].replace(["male","female"], [0, 1], inplace = True)
data["Fare"].fillna(np.mean(data["Fare"]), inplace = True)
data = data[["Survived","Pclass","Sex", "SibSp", "Parch", "Embarked","Fare"]]
data["Embarked"].fillna("S", inplace = True) #Sが多いので取り敢えずSで埋める
data["Embarked"].replace(["S","C", "Q"], [1,2,3], inplace=True)

Scaler1 = StandardScaler()

data_columns = data.columns
data = pd.DataFrame(Scaler1.fit_transform(data))
data.columns = data_columns

X = data.iloc[:train_len]
y = X["Survived"]
X = X.drop("Survived", axis = 1)
test = data.iloc[train_len:]
test.drop("Survived", axis = 1, inplace = True)

from torch import nn, optim
import torch
from torch.utils.data import TensorDataset, DataLoader
from statistics import mean
from torch.autograd import Variable




X_train, X_test, y_train, y_test =train_test_split(X, y, random_state = 10)

net = nn.Sequential(
    nn.Linear(6,20),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(20,20),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(20,2)
)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())
#損失関数のログ
train_losses = []
test_losses = []
correct_rate = []
num_right = []

X_train = torch.tensor(X_train.values, dtype = torch.float32)
y_train = torch.tensor(y_train.values, dtype = torch.float32)
X_test = torch.tensor(X_test.values, dtype = torch.float32)
y_test = torch.tensor(y_test.values, dtype = torch.float32)


#ミニバッチ用
ds = TensorDataset(X_train, y_train)
loader = DataLoader(ds, batch_size = 64, shuffle=True)

for epoc in range(200):
    running_loss = 0.0
    
    #trainモード
    net.train()
    for i,(xx, yy) in enumerate(loader):
        optimizer.zero_grad()
        y_pred = net(X_train)
        
        y_train = y_train.long()
        
        loss = loss_fn(y_pred,y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    
    #評価モード
    net.eval()
    h = net(X_test)
    y_test = y_test.long()
    
    loss = loss_fn(h,y_test)
    loss.backward()
    optimizer.step()
    test_losses.append(loss.item())
    
    values, labels = torch.max(h, 1)
    num_right.append(np.sum(labels.data.numpy() == y_test.numpy()))
    
    # 予測結果の確認 (yはFloatTensorなのでByteTensor
    # に変換してから比較する）
print(mean(num_right)/len(y_test))

plt.subplot()
plt.plot(train_losses)
plt.plot(test_losses, c="r")
plt.legend()

test = torch.tensor(test.values, dtype = torch.float32)

test_var = Variable(torch.FloatTensor(test), requires_grad=False) 
with torch.no_grad():
    test_result = net(test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()
submission = pd.DataFrame({'PassengerId': submission['PassengerId'], 'Survived': survived})
submission.to_csv('submission.csv', index=False)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True, p = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

# Part1
#mlp = nn.Sequential(
    #CustomLinear(6,200),
    #CustomLinear(200,200),
    #CustomLinear(200,200),
    #nn.Linear(200, 1))
#Part2
class MyMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln1 = CustomLinear(in_features, 20)
        self.ln2 = CustomLinear(20, 20)
        self.ln3 = CustomLinear(20, 20)
        self.ln4 = nn.Linear(20, out_features)
        
    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        return x

mlp = MyMLP(X_train.shape[1],2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr = 0.01)
batch_size = 64
n_epochs = 405
batch_no = len(X_train) // batch_size

train_loss = 0
train_loss_min = np.Inf

for epoch in range(n_epochs):
    mlp.train()
    for i in range(batch_no):
        start = i*batch_size
        end = start+batch_size
        x_var = Variable(torch.FloatTensor(X_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end])) 
        
        optimizer.zero_grad()
        output = mlp(x_var)
        loss = criterion(output,y_var)
        loss.backward()
        optimizer.step()
        
        values, labels = torch.max(output, 1)
        num_right = np.sum(labels.data.numpy() == y_train[start:end])
        train_loss += loss.item()*batch_size
    
    train_loss = train_loss / len(X_train)
    if train_loss <= train_loss_min:
        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
        torch.save(mlp.state_dict(), "model.pt")
        train_loss_min = train_loss
    
    if epoch % 200 == 0:
        print('')
        print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(y_train[start:end]) ))


plt.plot(train_losses)
X_test_var = Variable(torch.FloatTensor(test), requires_grad=False) 
with torch.no_grad():
    test_result = mlp(X_test_var)
values, labels = torch.max(test_result, 1)
survived = labels.data.numpy()
submission = pd.DataFrame({'PassengerId': submission['PassengerId'], 'Survived': survived})
submission.to_csv('submission_2.csv', index=False)





