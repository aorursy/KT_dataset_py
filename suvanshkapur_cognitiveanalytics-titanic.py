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
#importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#uploading dataset

df_1 = pd.DataFrame(pd.read_csv('../input/titanic/train.csv'))

df_2 = pd.DataFrame(pd.read_csv('../input/titanic/test.csv'))

df_3 =  pd.DataFrame(pd.read_csv('../input/titanic/gender_submission.csv'))
df_1.head()
#no. of nan values that dataset contains

for i in df_1.columns:

  print(i,"\t-\t", df_1[i].isna().mean()*100)
df_1 = df_1.drop(["Cabin"], axis=1)
df_1['Age'].fillna(df_1['Age'].median(), inplace=True)#Age Nan values

df_1['Embarked'].fillna(df_1['Embarked'].mode(), inplace=True)
df_1.info()
#im going to drop some non usable columns

df_1 = df_1.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)
#now preprocessing data using labelencoder lib

from sklearn.preprocessing import LabelEncoder



cat = df_1.drop(df_1.select_dtypes(exclude=['object']), axis=1).columns

print(cat)



enco = LabelEncoder()

df_1[cat[0]] = enco.fit_transform(df_1[cat[0]].astype('str'))



enco1 = LabelEncoder()

df_1[cat[1]] = enco1.fit_transform(df_1[cat[1]].astype('str'))
df_1.head()
df_1.info()
#visualize some of the columns that are targetted for pclass



sns.FacetGrid(df_1, col='Survived').map(plt.hist, 'Pclass')
#so from above graph we came to know that there is a good distribution in 3.0 and there are most of the classes havent surived in 3.0

# so we are going to predict for sex now

sns.FacetGrid(df_1, col='Survived').map(plt.hist, 'Sex')

#so by above graph we came to know that sex column is having nice corelation that how many male or female are survived

# now i am searching for age class

sns.FacetGrid(df_1, col='Survived').map(plt.hist, 'Age')
#so by this graph we came to know that how many in particular age group has survived or not

#now going to search in sibsp

sns.FacetGrid(df_1, col='Survived').map(plt.hist, 'SibSp')

#from above graph we are inferring how many are sibling or a spouse of each other like calculating a relation

#now im going to search for embarked

sns.FacetGrid(df_1, col='Survived').map(plt.hist, 'Embarked')
#from above graph we are searching that how many havent survived from particular place

#now predicting my train data

X = df_1.drop(['Survived'], axis=1)

y = df_1['Survived']
#now im splitting my data into test and train

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
#training model

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



df_4 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_4.head()
#plotting



plt.scatter([i for i in range(len(X_test["Age"]))], y_test, color='black')

plt.plot([i for i in range(len(X_test["Age"]))], y_pred, color='red')



plt.ylabel('Survived')

plt.xlabel('Passenger')



plt.show()
#checking Accuracy



from sklearn import metrics



# Generating roc curve using scikit-learn.

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

plt.plot(fpr, tpr)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.show()



print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))



# Accuracy scores

print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(y_pred, y_test)))
#uploading dataset

df_1 = pd.DataFrame(pd.read_csv('../input/titanic/train.csv'))

df_2 = pd.DataFrame(pd.read_csv('../input/titanic/test.csv'))

df_3 =  pd.DataFrame(pd.read_csv('../input/titanic/gender_submission.csv'))
#so above was the prediction for my train data

#now predicting for test data

df_2.head()
#columns for nan values 



for i in df_1.columns:

  print(i,"\t-\t", df_1[i].isna().mean()*100)
df_2 = df_2.drop(["Cabin"], axis=1)



df_2['Age'].fillna(df_2['Age'].median(), inplace=True) #filling Nan values of Age

df_1['Embarked'].fillna(df_1['Embarked'].mode(), inplace=True)
df_2.info()
#omitting some columns from the dataset which are not usable

PassengerId = df_2["PassengerId"]



df_2 = df_2.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)   #
df_2.head()
df_2[cat[0]] = enco.fit_transform(df_2[cat[0]].astype('str'))



df_2[cat[1]] = enco1.fit_transform(df_2[cat[1]].astype('str'))

y_test_pred = model.predict(df_2)
#plotting data graph



plt.scatter([i for i in range(len(df_2["Age"]))], y_test_pred, color='black')



plt.ylabel('Survived')

plt.xlabel('Passenger')



plt.show()
submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": y_test_pred

    })



submission.to_csv('./submission.csv', index=False)