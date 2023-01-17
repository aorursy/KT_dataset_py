# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#visualization

import matplotlib.pyplot as plt

%matplotlib inline

import math

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load the dataset

data=pd.read_csv('/kaggle/input/datasets_11657_16098_train.csv')
#check the shape of this dataset

data.shape
data
data.info()
data.head()
data.tail()
data.describe(include='all')
data.dtypes
data.isnull().sum()
#cleaning the null value

#As we can see, there are many null values under 'Cabin' column.

#687 out of 891 data points is a really high amount.and how to replace nan value with nalvale

# so i droped the Cabin column

data.drop(['Cabin'],axis=1,inplace=True)

data
#clean the age column 

data['Age'].fillna(data['Age'].mean(), inplace=True)
data
#clean the Embarked column

#replace the nan value with the top repeated element 'S'

data.Embarked.fillna(value="S",inplace=True)
data
data.isnull().sum()
#drop the unrelevent column 

data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
#countplot for Survived column

sns.countplot(data.Survived)

plt.show()
#Pclass has continous value so i can plot distplt

sns.distplot(data.Pclass[data.Survived==0])

sns.distplot(data.Pclass[data.Survived==1])

plt.legend(['Not Survived','Survived'])

plt.show()
data['Pclass'].unique()
sns.boxplot(x=data.Pclass,y=data.Age,data=data)

plt.show()
sns.countplot(data.Sex,hue=data.Survived)

plt.show()
sns.swarmplot(x=data.Sex,y=data.Fare,hue=data.Survived)

plt.show()

data['SibSp'].unique()
sns.countplot(data.SibSp,hue=data.Survived)

plt.show()
#not effect in our dataset so i can drop it

data.drop(['SibSp'],axis=1,inplace=True)
sns.countplot(data.Parch,hue=data.Survived)

plt.show()
sns.distplot(data.Fare[data.Survived==0])

sns.distplot(data.Fare[data.Survived==1])

plt.legend(['Not Survived','Survived'])

plt.show()
data["Fare"].plot.hist(figsize=(10,10),rwidth=0.97)

plt.show()
sns.countplot(data.Embarked,hue=data.Survived)

plt.show()
corr=data.corr()

plt.figure(figsize=(12,5))

sns.heatmap(corr,annot=True,cmap='coolwarm')

plt.show()
from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

data.Embarked=lb.fit_transform(data.Embarked)



le1=LabelEncoder()

data.Sex=le1.fit_transform(data.Sex)
data
#input and output selection

ip=data.drop(['Survived'],axis=1)

op=data['Survived']
data
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('Embarked',OneHotEncoder(),[5])],remainder='passthrough')



ip = np.array(ct.fit_transform(ip),dtype = np.str)

from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.4)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression()
#train the algorithm with the training data

alg.fit(xtr,ytr)

yp=alg.predict(xts)
from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)

print(cm)


accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)

precission=metrics.precision_score(yts,yp)

print(precission)





recall=metrics.recall_score(yts,yp)

print(recall)