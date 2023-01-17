# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#To load the dataset

cancer=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
##to print the instances

cancer.head()

cancer.columns
##To get the insight of dataset 

cancer.describe()

#to know the shape of dataset

cancer.shape
##for checking missing values

cancer.isnull().sum()
## Knowing missing values using visualisation

sns.heatmap(cancer.isnull(),cmap='coolwarm')
sns.distplot(cancer['radius_mean'])
sns.distplot(cancer['texture_mean'])
sns.distplot(cancer['perimeter_mean'])
sns.distplot(cancer['area_mean'])
#To drop id columns

cancer.drop(['id'],axis=1)
##Getting dummy parameter for the attribute diagnosis

m1=pd.get_dummies(cancer['diagnosis'],drop_first=True)
m1
rt=pd.concat([cancer,m1],axis=1)
rt.head()
##dropping some of the unwanted attributes

mt=rt.drop(['id','diagnosis','Unnamed: 32'],axis=1)
#printing some of the instances 

mt.head()
#considering all the attributes excluding class diagonosis

x=mt.drop(['M'],axis=1)
x.head()
y=mt['M']

y
##Dividing the dataset into training set and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)
X_train.head()
#fitting the model using the decision tree method

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
predicts=model.predict(X_test)
predicts
y_test
from sklearn.metrics import classification_report,confusion_matrix
##To print the confusion matrix

print(confusion_matrix(y_test,predicts))
#print classification report

print(classification_report(y_test,predicts))