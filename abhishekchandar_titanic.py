# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.shape
train.describe()
train.drop(labels='SibSp',axis = 1,inplace=True)

#train.head()



#since SibSp already deleted once, we can not find that column to perform 

#the above statements anymore.
train.info()
#filled age with median

imp_median = Imputer(missing_values = 'NaN',strategy = 'median')

imp_median.fit(train[['Age']])

train['Age'] = imp_median.transform(train[['Age']])
train.drop(labels=['Name','Ticket','Cabin','Pclass'],axis = 1,inplace=True)

train.info()
#get numerical column labels

numerical_features = list(train._get_numeric_data().columns)

print(numerical_features)

#get numerical column labels

categorical_features = list(set(train.columns) - set(train._get_numeric_data().columns))

print(categorical_features)

# Univariate Numerical Analysis

num = [ 'Age', 'Parch', 'Fare']

for i in num:

    plt.figure(figsize=(10,4))

    sns.distplot(train[i],bins=10)
#Multivariate analysis

sns.heatmap(train.corr(),cmap = 'viridis')

train.head()
#one hot encoding for train



def dummyEncode(train):

    le = LabelEncoder()

    for i in categorical_features:

        try:

            train[i] = le.fit_transform(train[i])

        except:

            print('Error encoding '+i)

    return train

train = dummyEncode(train)
test.drop(labels=['Name','Ticket','Cabin','SibSp'],axis = 1,inplace=True)

test.head()
#one hot encoing for test

#one hot encoding for train



def dummyEncode(train):

    le = LabelEncoder()

    for i in categorical_features:

        try:

            test[i] = le.fit_transform(test[i])

        except:

            print('Error encoding '+i)

    return test

test = dummyEncode(test)
imp_median = Imputer(missing_values = 'NaN',strategy = 'median')

imp_median.fit(test[['Age']])

test['Age'] = imp_median.transform(test[['Age']])



imp_median.fit(test[['Fare']])

test['Fare'] = imp_median.transform(test[['Fare']])
#Check!!!!!!! y_test doesnt exist, so obvio accuracy is 0.

X_train = train.iloc[:,2:-1]

y_train = train.iloc[:,1]



X_test = test.iloc[:,2:-1]

logistic_regressor = LogisticRegression()

logistic_regressor.fit(X_train,y_train)

y_pred = logistic_regressor.predict(X_test)
model = {'PassengerId':test['PassengerId'],'Survived':y_pred}

model = pd.DataFrame(model)

print(model)
model.to_csv('submission.csv',index = False)