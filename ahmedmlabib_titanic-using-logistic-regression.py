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
import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

import plotly.express as px

%matplotlib inline

cf.go_offline()
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.info()
test_data.info()
train_data.head()
train_data.isnull().sum().sort_values(ascending = False)
test_data.isnull().sum().sort_values(ascending = False)
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)
train_data.head()
px.box(x='Pclass',y='Age',data_frame=train_data.dropna())
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)
train_data.isnull().sum().sort_values(ascending = False)
test_data.isnull().sum().sort_values(ascending = False)
test_data['Fare'].ffill(inplace=True)
train_data['Embarked'].ffill(inplace=True)
train_data.info()
list_of_non_numeric_data=list(train_data.select_dtypes(include='object'))

list_of_non_numeric_data
train_data.drop('Ticket',axis=1,inplace=True)

test_data.drop('Ticket',axis=1,inplace=True)
train_data.info()
test_data.info()
def getTitles(name):

    name = str(name)

    title = name.split('.')[0]

    title = title.split(',')

    return title[1]
train_data['Title'] = train_data['Name'].apply(getTitles)

train_data['Title']
test_data['Title'] = test_data['Name'].apply(getTitles)

test_data['Title']
def cleanTitle(title):

    if title in [' Mr',' Mrs',' Master',' Miss']:

        return title

    else:

        return "Others"
train_data['Title'] = train_data['Title'].apply(cleanTitle)

test_data['Title'] = test_data['Title'].apply(cleanTitle)
train_data.head()
Title_train = pd.get_dummies(train_data['Title'],drop_first=True)

Title_test = pd.get_dummies(test_data['Title'],drop_first=True)

sex_train = pd.get_dummies(train_data['Sex'],drop_first=True)

embark_train = pd.get_dummies(train_data['Embarked'],drop_first=True)

sex_test = pd.get_dummies(test_data['Sex'],drop_first=True)

embark_test = pd.get_dummies(test_data['Embarked'],drop_first=True)



train_data.drop(['Sex','Embarked','Name','Title'],axis=1,inplace=True)

test_data.drop(['Sex','Embarked','Name','Title'],axis=1,inplace=True)
train_data=pd.concat([train_data,sex_train,embark_train,Title_train],axis=1)

test_data=pd.concat([test_data,sex_test,embark_test,Title_test],axis=1)
test_data.head(10)
from sklearn.preprocessing import MinMaxScaler
X_train=train_data.drop(['Survived','PassengerId'],axis=1)

y_train= train_data['Survived']

X_test=test_data.drop(['PassengerId'],axis=1)
Scaler=MinMaxScaler()
X_train = Scaler.fit_transform(X_train)

X_test = Scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=10000)

logmodel.fit(X_train,y_train)
test_data['Survived']=logmodel.predict(X_test)
test_data[['PassengerId', 'Survived']].to_csv('kaggle_submission.csv', index = False)