# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns           

import matplotlib.pyplot as plt

%matplotlib inline



# warning ignore

import warnings 

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

data.head()
data.isna().sum() #for checking null values
sum(data.duplicated()) #checking if any duplicate values are there are not
data.info() #checking about the data type 
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Geography'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by countries')

ax[0].set_ylabel('count')

sns.countplot(data=data,x='Geography',hue='Exited',ax=ax[1])

ax[1].set_title('Countries:Exited vs Non Exited')

ax[1].set_ylabel('count');
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Gender'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by gender')

ax[0].set_ylabel('count')

sns.countplot(data=data,x='Gender',hue='Exited',ax=ax[1])

ax[1].set_title('Gender:Exited vs Non Exited')

ax[1].set_ylabel('count');
Non_Exited = data[data['Exited']==0]

Exited = data[data['Exited']==1]



plt.subplots(figsize=(18,8))

sns.distplot(Non_Exited['Age'])

sns.distplot(Exited['Age'])

plt.title('Age:Exited vs Non Exited')

plt.legend([0,1],title='Exited')

plt.ylabel('percentage');

f,ax = plt.subplots(1,2,figsize=(18,8))

data['NumOfProducts'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of customer by Number of Product')

ax[0].set_ylabel('count')

sns.countplot(data=data,x='NumOfProducts',hue='Exited',ax=ax[1])

ax[1].set_title('Number of Product:Exited vs Non Exited')

ax[1].set_ylabel('count');
plt.figure(figsize=(18,8))

plt.hist(x='CreditScore',bins=100,data=Non_Exited,edgecolor='black',color='red')

plt.hist(x='CreditScore',bins=100,data=Exited,edgecolor='red',color='black')

plt.title('Credit score: Exited vs Non-Exited')

plt.legend([0,1],title='Exited');
plt.figure(figsize=(18,8))

p1=sns.kdeplot(Non_Exited['Balance'], shade=True, color="r")

p1=sns.kdeplot(Exited['Balance'], shade=True, color="b");

plt.title('Account Balance: Exited vs Non-Exited')

plt.legend([0,1],title='Exited');
plt.title("features correlation matrics".title(),

          fontsize=20,weight="bold")



sns.heatmap(data.corr(),annot=True,cmap='RdYlBu',linewidths=0.2, vmin=-1, vmax=1,linecolor = 'black') 

fig=plt.gcf()

fig.set_size_inches(10,8);
X=data.iloc[:,3:13]    #features

y=data['Exited']     #label
X.head()
from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score
#encoding and labelling the necessary columns

label=LabelEncoder()

scaler = StandardScaler()

X['Geography']=label.fit_transform(X['Geography'])

X['Gender']=label.fit_transform(X['Gender'])

X[['CreditScore','Balance','EstimatedSalary']]=scaler.fit_transform(X[['CreditScore','Balance','EstimatedSalary']])

X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state = 100) #training and testing
per=Perceptron()
per.fit(X_train,y_train)
y_pred_train=per.predict(X_train)

y_pred_test=per.predict(X_test)
print("Training Accuracy: ",accuracy_score(y_pred_train,y_train))

print("Testing Accuracy: ",accuracy_score(y_pred_test,y_test))
param_grid={'eta0': [1.0,0.5,1e-10], 'max_iter': [5,10,20,30,40,50]}
grid=GridSearchCV(per, param_grid, cv=100)
grid.fit(X_train,y_train)
grid.best_score_
grid.best_params_
perceptron=grid.best_estimator_

y_pred_train=perceptron.predict(X_train)

y_pred_test=perceptron.predict(X_test)

print("Training Accuracy: ",accuracy_score(y_pred_train,y_train))

print("Testing Accuracy: ",accuracy_score(y_pred_test,y_test))