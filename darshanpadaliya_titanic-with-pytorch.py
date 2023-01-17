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
!pip install seaborn --upgrade
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
#### Removing unnecessary columns



df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

df.info()
#### Handling null values



df['Embarked'].fillna(df['Embarked'].mode()[0],inplace =True)

df['Age'].fillna(df['Age'].median(),inplace =True)

df.info()
#### Relation of every feature with Survived



fig, a = plt.subplots(3, 2, figsize=(16, 20))

fig.delaxes(a[2][1])

sns.countplot(x='Pclass',data=df,hue='Survived',ax=a[0,0],palette=['darkgray','paleturquoise'])

sns.countplot(x='Sex',data=df,hue='Survived',ax=a[0,1],palette=['darkgray','paleturquoise'])

sns.countplot(x='SibSp',data=df,hue='Survived',ax=a[1,0],palette=['darkgray','paleturquoise'])

sns.countplot(x='Parch',data=df,hue='Survived',ax=a[1,1],palette=['darkgray','paleturquoise'])

sns.countplot(x='Embarked',data=df,hue='Survived',ax=a[2,0],palette=['darkgray','paleturquoise'])

sns.displot(data=df,x='Age',hue='Survived',element="step",aspect=3,palette=['darkgray','paleturquoise'])

sns.displot(data=df,x='Fare',hue='Survived',element="step",aspect=3,palette=['darkgray','paleturquoise'])
X = df.iloc[:,1:]

y = df.iloc[:,0]
#### Catagorical Features



cat_feat = X.select_dtypes(include='object').columns.tolist()

cat_feat
#### Numerical Feature



num_feat = list(set(X.columns.tolist()) - set(cat_feat))

num_feat
#### Encoding Catagorical Features



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='if_binary',sparse=False)

cat = ohe.fit_transform(X[cat_feat])

ohe.get_feature_names()
df_C = pd.DataFrame(data=cat, columns=ohe.get_feature_names())

df_C.head()
X = pd.concat([X, df_C], axis=1)

X.drop(cat_feat, axis=1, inplace=True)

X.head()
#### Train Test split



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
import torch

import torch.nn as nn

import torch.nn.functional as F
#### Converting dataframe to array 



X_train = X_train.values    

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values





#### Converting array into tensors



X_train=torch.FloatTensor(X_train)

X_test=torch.FloatTensor(X_test)

y_train=torch.LongTensor(y_train)

y_test=torch.LongTensor(y_test)

#### Creating Modelwith Pytorch



class ANN_Model(nn.Module):

    def __init__(self,input_features=9,hidden1=30,hidden2=30,hidden3=30,out_features=2):

        super().__init__()

        self.f_connected1=nn.Linear(input_features,hidden1)

        self.f_connected2=nn.Linear(hidden1,hidden2)

        self.f_connected3=nn.Linear(hidden2,hidden3)

        self.out=nn.Linear(hidden3,out_features)

    def forward(self,x):

        x=F.leaky_relu(self.f_connected1(x))

        x=F.leaky_relu(self.f_connected2(x))

        x=torch.sigmoid(self.f_connected3(x))

        x=self.out(x)

        return x
#### Instantiating ANN_model



torch.manual_seed(20)

model=ANN_Model()
model.parameters
#### For Backward Propogation-- Defining the loss_function and the optimizer



loss_function=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
#### Training the Model



epochs=1000

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
#### ploting the loss function



plt.figure(figsize=(20,8))

plt.plot(range(epochs),final_losses)

plt.ylabel('Loss')

plt.xlabel('Epoch')
#### Prediction In X_test data



predictions=[]

with torch.no_grad():

    for i,data in enumerate(X_test):

        y_pred=model(data)

        predictions.append(y_pred.argmax().item())

        print(y_pred.argmax().item())

        
#### Confusion matrix



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,predictions)

plt.figure(figsize=(10,6))

sns.heatmap(cm,annot=True)

plt.xlabel('Actual Values')

plt.ylabel('Predicted Values')
#### Accuracy Score



from sklearn.metrics import accuracy_score

score=accuracy_score(y_test,predictions)

print('accuracy : ',score*100,'%') 