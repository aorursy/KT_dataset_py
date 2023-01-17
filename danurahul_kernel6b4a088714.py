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
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
df.isnull().sum()
import seaborn as sns
sns.pairplot(df,hue="Outcome")
from sklearn.model_selection import train_test_split
#X=df.iloc[:,:-1]
#y=df.iloc[:,-1]
X=df.drop('Outcome',axis=1).values
y=df['Outcome'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#import libraries from pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
df.shape
#### Creating Modelwith Pytorch
class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x
####instantiate my ANN_model
torch.manual_seed(20)
model=ANN_Model()
##backward propogation, define loss_function , optim,
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=500
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss)
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
#prediction on x_test 
predictions=[]
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(predictions,y_test)
cm
plt.figure(figsize=(10,4))
sns.heatmap(cm,annot=True)
plt.xlabel('action')
plt.ylabel('predicted values')
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,predictions)
score
#save and load model
torch.save(model,'diabetes.pt')
model=torch.load('diabetes.pt')
model.eval()
#prediction of new data point
list(df.iloc[0,:-1])

#new data
list1=[6.0, 130.0, 72.0, 35.0, 0.0, 35.6, 0.600, 50.0]
new_data=torch.tensor(list1)
#predict new data using pytoch
with torch.no_grad():
    
    print(model(new_data).argmax().item())
