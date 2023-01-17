# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings 

warnings.filterwarnings ('ignore')

titanic_data = pd.read_csv('../input/train.csv')
titanic_data.head(10)

print("# of passengers in original data:"  +str(len(titanic_data.index)))
sns.countplot(x="Survived",data=titanic_data)
sns.countplot(x="Survived", hue="Sex", data=titanic_data)
sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
sns.countplot(x="SibSp", data=titanic_data)
# now lets understand the data distribution using Histogrmas
titanic_data['Age'].plot.hist()
# in above figure it is little bit difficult to see clearly, lets incearse its size
titanic_data["Age"].plot.hist(bins=10, figsize=(10,5)) # you can adjust the graph size and bin used for bar width
titanic_data.info()
titanic_data.isnull().sum()
titanic_data.dropna(inplace=True)
titanic_data.isnull().sum() # after appliying dropna  function we droped all Na valued columns.
pd.get_dummies(titanic_data["Sex"])
sex= pd.get_dummies(titanic_data["Sex"], drop_first=True)
sex.head(5)
embark= pd.get_dummies(titanic_data["Embarked"], drop_first=True)
embark.head(5)
Pcl= pd.get_dummies(titanic_data["Pclass"], drop_first=True)
Pcl.head(5)
titanic_data=pd.concat([titanic_data,sex,embark,Pcl], axis=1) # Here adding is Successful
titanic_data.head(5)
titanic_data.drop(['PassengerId','Name','Ticket', 'Embarked','Sex'], axis=1, inplace=True)
titanic_data.head()
titanic_data.drop('Pclass', axis=1, inplace=True)
titanic_data.drop('Cabin', axis=1, inplace=True)
titanic_data.head(5)
x= titanic_data.drop(["Survived"],axis=1) # independent variable (FEATURES). Rest all columns wii be considered as training

                                           # data except Survived, because we need to predict it

y= titanic_data["Survived"]  #Dependent variable # This is TARGET column that we have to predict using ML model.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(x_train,y_train)
y_predicted =logmodel.predict(x_test)
logmodel.predict(x_test)
logmodel.score(x_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_predicted, y_test)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



titanic_data.to_csv(filename,index=False)



print('Saved file: ' + filename)