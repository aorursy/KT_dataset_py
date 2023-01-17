import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df =  pd.read_csv("/kaggle/input/titanic/train.csv",index_col=0)

df.head(20)
len(df.index)
del df['Name']
df.isnull().sum(axis = 0)
plt.figure(figsize=(10,5))

sns.heatmap(df.isnull(), yticklabels=False,cbar=False)
del df['Cabin']
df.loc[df["Sex"] == "male","Sex"] = 0

df.loc[df["Sex"] == "female","Sex"] = 1

df.head()
df['Embarked'].value_counts() 
df['Embarked'] = df.Embarked.fillna('S')
g = sns.FacetGrid(df, col="Embarked")

#g = sns.FacetGrid(df, col="Embarked", row = 'Sex')

g = g.map(plt.hist, "Survived")
pd.crosstab(df.Embarked,df.Survived)
df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)

df.head()
del df['Ticket']

del df['Parch']

del df['SibSp']
df.head()
keys = df.columns



for i in keys[1:]:

    g = sns.FacetGrid(df, col="Survived", height=3, aspect=2)

    g.map(plt.hist, i)
meanAge = df['Age'].mean()

print(meanAge)
plt.figure(figsize=(10,5))

sns.heatmap(df.isnull(), yticklabels=False,cbar=False)
df['Age'] = df.Age.fillna(meanAge)

df['Age'] = df['Age'].astype('int64')

df.head()
df.loc[df["Age"] < 15,"Age"] = 0

df.loc[(df["Age"] >= 15) & (df["Age"] < 30),"Age"] = 1

df.loc[(df["Age"] >= 30) & (df["Age"] < 45),"Age"] = 2

df.loc[(df["Age"] >= 45) & (df["Age"] < 60),"Age"] = 3

df.loc[(df["Age"] >= 60), 'Age'] = 4

df.head()
m = sns.FacetGrid(df, col='Survived')

m.map(plt.hist, 'Age', bins=20)
df.loc[df["Fare"] < 10,"Fare"] = 0

df.loc[(df["Fare"] >= 10) & (df["Fare"] < 25),"Fare"] = 1

df.loc[(df["Fare"] >= 25) & (df["Fare"] < 50),"Fare"] = 2

df.loc[(df["Fare"] >= 50) & (df["Fare"] < 100),"Fare"] = 3

df.loc[(df["Fare"] >= 100), 'Fare'] = 4

df.head()
m = sns.FacetGrid(df, col='Survived')

m.map(plt.hist, 'Fare', bins=20)
df.nunique()
df.dtypes
mean = df['Age'].mean()

df['Age'] = df['Age'].fillna(mean)
sns.heatmap(df.corr(), annot = True)
y=df["Survived"]

x=df.iloc[:,1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=4130)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rdclass=RandomForestClassifier()

rdclass.fit(X_train,y_train)

ypred=rdclass.predict(X_test)



accRFC = accuracy_score(y_test,ypred)

print(accRFC)
from sklearn import svm

clf = svm.SVR()

clf.fit(X_train,y_train)



ypred = clf.predict(X_test)

accSVM_D = accuracy_score(y_test, ypred.round())

print(accSVM_D)
val = []

euc = []



for c in range(1,50):

    for eps in range(0,10):

        clf = svm.SVR(C=c, epsilon=(eps/10))

        clf.fit(X_train,y_train)



        ypred = clf.predict(X_test)

        euc.append([c,(eps/10)])

        val.append([accuracy_score(y_test, ypred.round()),c,eps])



accSVM_A = max(val)

accSVM_A = accSVM_A[0]

print(accSVM_A)
max(val)
clf = svm.SVR(C=49, epsilon=(4/10))

clf.fit(X_train,y_train)



ypred = clf.predict(X_test)

accuracy_score(y_test, ypred.round())
ypred = pd.DataFrame(ypred)

ypred = pd.Series.round(ypred)

ypred = ypred.astype('int64')

accuracy_score(y_test, ypred.round())
import numpy as np

x_ntest = np.array(X_test)

y_ntest = np.array(y_test)

x_ntrain = np.array(X_train)

y_ntrain = np.array(y_train)
print(x_ntrain.shape, y_ntrain.shape)
from xgboost import XGBClassifier



model = XGBClassifier()



# fit the model with the training data

model.fit(x_ntrain,y_ntrain)



# predict the target on the train dataset

predict_train = model.predict(x_ntrain)



# Accuray Score on train dataset

accuracy_train = accuracy_score(y_ntrain,predict_train)

print('accuracy_score on train dataset : ', accuracy_train)



# predict the target on the test dataset

predict_test = model.predict(x_ntest)



# Accuracy Score on test dataset

accuracy_test = accuracy_score(y_ntest,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



# make datasets

dtrain = xgb.DMatrix(x_ntrain, label=y_train)

dtest = xgb.DMatrix(x_ntest)



# set up

param = {'max_depth':8, 'eta':0.1, 'objective':'binary:hinge' }

num_round = 10



# fit the model with the training data

bst = xgb.train(param, dtrain, num_round)



# make prediction

preds = bst.predict(dtest)

preds = preds.astype('int64')



# Accuracy Score on test dataset

accuracy_test = accuracy_score(y_test,preds)

print('accuracy_score on test dataset : ', accuracy_test)
df =  pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)



#del Values

del df['Cabin']

del df['Name']

del df['Ticket']

del df['Parch']

del df['SibSp']



#fillup values

df['Age'] = df.Age.fillna(meanAge)

df['Embarked'] = df.Embarked.fillna('S')

df['Fare'] = df.Fare.fillna(0.0)



#cast values

df['Age'] = df['Age'].astype('int64')



#map values

df.loc[df["Sex"] == "male","Sex"] = 0

df.loc[df["Sex"] == "female","Sex"] = 1



df.loc[df["Age"] < 15,"Age"] = 0

df.loc[(df["Age"] >= 15) & (df["Age"] < 30),"Age"] = 1

df.loc[(df["Age"] >= 30) & (df["Age"] < 45),"Age"] = 2

df.loc[(df["Age"] >= 45) & (df["Age"] < 60),"Age"] = 3

df.loc[(df["Age"] >= 60), 'Age'] = 4



df.loc[df["Fare"] < 10,"Fare"] = 0

df.loc[(df["Fare"] >= 10) & (df["Fare"] < 25),"Fare"] = 1

df.loc[(df["Fare"] >= 25) & (df["Fare"] < 50),"Fare"] = 2

df.loc[(df["Fare"] >= 50) & (df["Fare"] < 100),"Fare"] = 3

df.loc[(df["Fare"] >= 100), 'Fare'] = 4



df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)



df.head()

df.isna().sum()
data = np.array(df)
dfinal = xgb.DMatrix(data)
#preds = bst.predict(dfinal)



#my_submission = pd.DataFrame({'PassengerId': df.index, 'Survived': preds})





ypred = clf.predict(df)



ypred = pd.DataFrame(ypred)

ypred = pd.Series.round(ypred)

ypred

my_submission = pd.DataFrame({'PassengerId': df.index})

my_submission['Survived'] = ypred



my_submission['Survived'] = my_submission['Survived'].astype('int64')

my_submission.to_csv('submission.csv', index=False)

my_submission.head(20)