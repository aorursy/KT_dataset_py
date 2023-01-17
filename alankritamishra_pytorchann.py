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
data= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data['Outcome'])
def boxplot(df: pd.DataFrame) -> None:
    """
    Visualize a boxplot for each feature for each class.
    """
    fig, axis = plt.subplots()

    for col in df.columns:
        if col != 'Outcome':
            sns.boxplot(x='Outcome', y=col, data=df, palette='Blues')
            plt.show()
            
# call the function to display the boxplot
boxplot(df=data)
sns.scatterplot(x='Insulin', y='Glucose', hue='Outcome', data=data)
plt.xlabel('Insulin')
plt.ylabel('Glucose')
#plotting age feature
sns.FacetGrid(data, hue="Outcome", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend()
plt.show()
sns.pairplot(data,hue='Outcome')
data.corr()
plt.subplots(figsize=(15,8))

sns.heatmap(data.corr(),annot=True,cmap='rainbow')
y= data['Outcome']
data.drop(['Outcome'],axis=1,inplace=True)
x= data
x.head()
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaled= scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val= train_test_split(scaled,y)
#importing torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
X_train= torch.FloatTensor(X_train)
X_val= torch.FloatTensor(X_val)

y_train= torch.LongTensor(y_train.values)
y_val= torch.LongTensor(y_val.values)
class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=25,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,h):
        h=F.relu(self.f_connected1(h))
        h=F.relu(self.f_connected2(h))
        h=self.out(h)
        return h
torch.manual_seed(20)
model=ANN_Model()
model.parameters
loss_function= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(),lr=0.01)
epochs=100
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
plt.xlabel('Epochs')
plt.show()
#predictions in data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_val):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
predictions
from sklearn.metrics import accuracy_score
accuracy_score= accuracy_score(y_val,predictions)
print("The accuracy of your model is {} percent".format(accuracy_score*100))
