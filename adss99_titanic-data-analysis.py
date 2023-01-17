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
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head()
train_data.drop(['Name','Ticket','Cabin'],axis=1,inplace = True)

train_data.set_index('PassengerId',inplace = True)

avg_age = train_data['Age'].astype('float').mean(axis = 0 )

train_data['Age'].replace(np.nan,avg_age,inplace = True)

train_data.dropna(subset = ['Embarked'],axis = 0 ,inplace = True)

train_data.head()
sex_dummies = pd.get_dummies(train_data['Sex'])

train_data = pd.concat([train_data,sex_dummies],axis = 1)

train_data.drop(['Sex'],axis=1 ,inplace = True)

Embarked_Dummies = pd.get_dummies(train_data["Embarked"])

train_data = pd.concat([train_data,Embarked_Dummies],axis=1)

train_data.drop(['Embarked'],axis=1,inplace=True)

train_data['Age'] = train_data['Age'].astype(int, copy = True)

train_data['Fare'] = train_data['Fare'].astype(int,copy=True)

train_data.head()
%matplotlib inline

import matplotlib as plt

from matplotlib import pyplot
plt.pyplot.hist(train_data["Age"])



# set x/y labels and plot title

plt.pyplot.xlabel("Age")

plt.pyplot.ylabel("count")

plt.pyplot.title("Age bins")
bins = np.linspace(min(train_data["Age"]), max(train_data["Age"]), 9)

group_names = ['0-10', '11-20', '21-30','31-40','41-50','51-60','61-70','71-80']

train_data['Age_binned'] = pd.cut(train_data['Age'], bins, labels=group_names, include_lowest=True )

Age_dummies = pd.get_dummies(train_data['Age_binned'])

train_data = pd.concat([train_data,Age_dummies],axis = 1)

train_data.drop(['Age','Age_binned'],axis=1 ,inplace = True)

train_data.head()
plt.pyplot.hist(train_data["Fare"])



# set x/y labels and plot title

plt.pyplot.xlabel("fare")

plt.pyplot.ylabel("count")

plt.pyplot.title("fare bins")
bins = np.linspace(min(train_data["Fare"]), max(train_data["Fare"]),11)

Fare_names = ['0-50', '51-100', '101-150','151-200','201-250','251-300','300-350','350-400','401-450','450-500']

train_data['Fare_binned'] = pd.cut(train_data['Fare'], bins, labels=Fare_names, include_lowest=True )

Fare_dummies = pd.get_dummies(train_data['Fare_binned'])

train_data = pd.concat([train_data,Fare_dummies],axis = 1)

train_data.drop(['Fare','Fare_binned'],axis=1 ,inplace = True)

train_data.head()

test_data.head()
test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace = True)

test_data.set_index('PassengerId',inplace = True)

avg_age = test_data['Age'].astype('float').mean(axis = 0 )

test_data['Age'].replace(np.nan,avg_age,inplace = True)

test_data.dropna(subset = ['Embarked'],axis = 0 ,inplace = True)

sex_dummies = pd.get_dummies(test_data['Sex'])

test_data = pd.concat([test_data,sex_dummies],axis = 1)

test_data.drop(['Sex'],axis=1 ,inplace = True)

Embarked_Dummies = pd.get_dummies(test_data["Embarked"])

test_data = pd.concat([test_data,Embarked_Dummies],axis=1)

test_data.drop(['Embarked'],axis=1,inplace=True)

test_data.head()

avg_fare = test_data['Fare'].astype('float').mean(axis = 0 )

test_data['Fare'].replace(np.nan,avg_fare,inplace = True)

test_data['Age'] = test_data['Age'].astype(int, copy = True)

test_data['Fare'] = test_data['Fare'].astype(int,copy=True)

test_data.head()
bins = np.linspace(min(test_data["Age"]), max(test_data["Age"]), 9)

group_names = ['0-10', '11-20', '21-30','31-40','41-50','51-60','61-70','71-80']

test_data['Age_binned'] = pd.cut(test_data['Age'], bins, labels=group_names, include_lowest=True )

Age_dummies = pd.get_dummies(test_data['Age_binned'])

test_data = pd.concat([test_data,Age_dummies],axis = 1)

test_data.drop(['Age','Age_binned'],axis=1 ,inplace = True)

test_data.head()
bins = np.linspace(min(test_data["Fare"]), max(test_data["Fare"]),11)

Fare_names = ['0-50', '51-100', '101-150','151-200','201-250','251-300','300-350','350-400','401-450','450-500']

test_data['Fare_binned'] = pd.cut(test_data['Fare'], bins, labels=Fare_names, include_lowest=True )

Fare_dummies = pd.get_dummies(test_data['Fare_binned'])

test_data = pd.concat([test_data,Fare_dummies],axis = 1)

test_data.drop(['Fare','Fare_binned'],axis=1 ,inplace = True)

test_data.head()

train_df = train_data.drop(['Survived'],axis=1)

train_df.head()
X_train = np.asarray(train_df)

X_train[0:5]



X_train.shape
Y_train =np.asarray(train_data['Survived'])

Y_train[0:5]

Y_train.shape
X_test =np.asarray(test_data)

X_test[0:5]
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)

LR
Y_test = LR.predict(X_test)

Y_test[0:5]
Y_prob = LR.predict_proba(X_test)

Y_prob[0:5]
PassengerId = test_data.index
df = pd.DataFrame(PassengerId)

df['Survived'] = Y_test

df
df.to_csv('csv_to_submit.csv', index = False)
 