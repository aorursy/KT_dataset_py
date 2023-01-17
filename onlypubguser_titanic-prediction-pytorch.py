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
import torch

from torch import nn,optim

import torch.nn.functional as F

import torchvision

from torchvision import datasets as datasets, models ,transforms

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import pandas as pd

#from collections import OrderedDict

import time

import csv
titanic_train=pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_test=pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_train
titanic_test
def Sex2num(data):

  if data=='female':

    return 1

  else:

    return 0
def Embarked2num(data):

  if data=='S':

    return 0

  elif data=='Q':

    return 1

  else:

    return 2
titanic_train['Embarked']=titanic_train['Embarked'].apply(Embarked2num)

titanic_test['Embarked']=titanic_test['Embarked'].apply(Embarked2num)
titanic_train['Sex']=titanic_train['Sex'].apply(Sex2num)

titanic_test['Sex']=titanic_test['Sex'].apply(Sex2num)
titanic_train
titanic_test
FOB=titanic_train['SibSp']+titanic_train['Parch']

titanic_train['FOB']=FOB

FOB=titanic_test['SibSp']+titanic_test['Parch']

titanic_test['FOB']=FOB
titanic_train=titanic_train.drop(['Name','PassengerId','Ticket','Cabin','SibSp','Parch'],axis='columns')

passenger=titanic_test['PassengerId']

titanic_test=titanic_test.drop(['Name','PassengerId','Ticket','Cabin','SibSp','Parch'],axis='columns')
titanic_train['Fare']=titanic_train['Fare'].fillna(titanic_train['Fare'].median())

titanic_train['Age']=titanic_train['Age'].fillna(titanic_train['Age'].median())

titanic_train['Embarked']=titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode())

titanic_train
titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())

titanic_test['Age']=titanic_test['Age'].fillna(titanic_test['Age'].median())

titanic_test['Embarked']=titanic_test['Embarked'].fillna(titanic_test['Embarked'].mode())
survived=titanic_train['Survived']

titanic_train=titanic_train.drop(['Survived'],axis='columns')

titanic_train
titanic_tensor_train=torch.Tensor(titanic_train[:].values)

survived_tensor_train=torch.Tensor(survived[:].values)

titanic_tensor_test=torch.Tensor(titanic_test[:].values)
print(titanic_tensor_train.shape)

print(survived_tensor_train.shape)

print(titanic_tensor_test.shape)

print(titanic_tensor_train[1:5])
class Everything(nn.Module):

  

  def __init__(self):

      super(Everything, self).__init__()

      self.hidden1 = nn.Linear(6, 320)

      self.hidden2 = nn.Linear(320, 2)

      self.dropout=nn.Dropout(p=0.2)

  def forward(self, x):

      x = self.hidden1(x)

      x = F.tanh(x)

      x = self.dropout(x)

      x = self.hidden2(x)

      x = torch.sigmoid(x)

      x = F.softmax(x)

      

      return x
model=Everything()
optimizer=optim.Adam(model.parameters(),lr=0.01)

criterion = nn.CrossEntropyLoss()
loss_at_epoch_training=np.array([])
epochs=50

batch=40

batch_num=len(titanic_tensor_train)/batch

print(batch_num)

batch_num=int(batch_num)

print(batch_num)
for i in range(epochs):

  print(i)

  training_loss=0

  # testing_loss=0

  start_time=time.time()

  for j in range(batch_num):

    start=j*batch



    end=start+batch



    in_f = titanic_tensor_train[start:end]



    out_f = survived_tensor_train[start:end]



    optimizer.zero_grad()



    Probs=model(in_f)



    out_f=out_f.long()



    loss=criterion(Probs.view(40,2),out_f.view(40))



    loss.backward()



    optimizer.step()



    training_loss+=loss



  end_time=time.time()

  print("Elapsed time for training epoch:",(end_time-start_time))

  training_loss=training_loss/len(titanic_tensor_train)

  print("training_loss",training_loss)

  loss_at_epoch_training=np.append(loss_at_epoch_training,training_loss.detach().numpy())

  torch.save(model.state_dict,'titanic.pth')
plt.plot(loss_at_epoch_training[:], label="training")

plt.show()
submission=np.array([])
with torch.no_grad():

  output=model(titanic_tensor_test)

output=output.numpy()

for i in range(len(output)):

  if output[i][0]>output[i][1]:

    submission=np.append(submission,0)

  else:

    submission=np.append(submission,1)
submission
submission.mean()
submission=submission.astype(int)
passenger
submission_df=pd.DataFrame([])
submission_df['PassengerId']=passenger

submission_df['Survived']=submission
submission_df
submission_df.to_csv('submission_by_OnlyPubgUser.csv', index=False)