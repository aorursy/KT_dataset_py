# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/black-friday-sales-prediction/train.csv")

test = pd.read_csv("/kaggle/input/black-friday-sales-prediction/test.csv")
train.head()
sns.countplot(train['Age'],hue=train['City_Category'])
sns.heatmap(train.corr(),annot=True)
sns.countplot(train['Occupation'],hue=train['Marital_Status'])
sns.countplot(train['Age'],hue=train['Marital_Status'])
sns.countplot(train['Purchase'].value_counts()[:15],hue=train['Marital_Status'])
def chanegAge(x):

    if(x=="0-17"):

        return 0

    elif(x=="18-25"):

        return 1

    elif(x=="26-35"):

        return 2

    elif(x=="36-45"):

        return 3

    elif(x=="46-50"):

        return 4

    elif(x=="51-55"):

        return 5

    elif(x=="55+"):

        return 6
enc = LabelEncoder()

train['Gender'] = enc.fit_transform(train['Gender'])

train['Occupation'] = enc.fit_transform(train['Occupation'])

train['City_Category'] = enc.fit_transform(train['City_Category'])

train['Stay_In_Current_City_Years'] = enc.fit_transform(train['Stay_In_Current_City_Years'])

train['Product_Category_1'] = enc.fit_transform(train['Product_Category_1'])

train['Product_Category_2'] = enc.fit_transform(train['Product_Category_2'])

train['Product_Category_3'] = enc.fit_transform(train['Product_Category_3'])

train['Age'] = train['Age'].apply(chanegAge)
train.head()

train = train[:10000]

train.shape
train.drop(['User_ID','Product_ID'],axis=1,inplace=True)

X = train.drop(['Purchase'],axis=1)

y = train['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
log = DecisionTreeClassifier()

log.fit(X_train,y_train)

predict = log.predict(X_test)

accuracy_score(predict,y_test)
log = LogisticRegression()

log.fit(X_train,y_train)

predict = log.predict(X_test)

accuracy_score(predict,y_test)