# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import os

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import xgboost

from xgboost import XGBRegressor

import lightgbm as lgb

import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
print(df_train.shape)

print(df_test.shape)
df_train.isnull().sum()
df_train.describe()
df_train.dtypes
df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S',inplace=True)

df_test['Embarked'].fillna('S',inplace=True)
df_train['Cabin'].fillna('None',inplace=True)

df_test['Cabin'].fillna('None',inplace=True)
sns.countplot(df_train['Age'])
age=df_train['Age'].unique()
print(age)
age.sort()
age=np.delete(age,-1)
age
mean=age.mean()
df_train['Age'].fillna(mean,inplace=True)

df_test['Age'].fillna(mean,inplace=True)
fare=df_test['Fare'].unique()
fare.sort()
fare=np.delete(age,-1)
fare=fare.mean()

df_test['Fare'].fillna(fare,inplace=True)
print(df_test.isnull().sum())

print('---------------------------')

print(df_train.isnull().sum())
bins = [0,10,20,30,40,50,60,70,80]

labels = [1,2,3,4,5,6,7,8]

df_train['Agebinned'] = pd.cut(df_train['Age'], bins=bins, labels=labels)

df_test['Agebinned']=pd.cut(df_test['Age'],bins=bins,labels=labels)
df_train.head()
Embarkeddum=pd.get_dummies(df_train['Embarked'])

Embarkeddumt=pd.get_dummies(df_test['Embarked'])

df_train=pd.concat([df_train,Embarkeddum],axis=1)

df_test=pd.concat([df_test,Embarkeddumt],axis=1)

Pclassdum=pd.get_dummies(df_train['Pclass'])

Pclassdumt=pd.get_dummies(df_test['Pclass'])

Pclassdum.rename(columns={1:'p1',2:'p2',3:'p3'},inplace=True)

Pclassdumt.rename(columns={1:'p1',2:'p2',3:'p3'},inplace=True)

df_train=pd.concat([df_train,Pclassdum],axis=1)

df_test=pd.concat([df_test,Pclassdumt],axis=1)
df_train.dtypes
bins = [-1,0,100,200,300,400,500,600]

labels = [0,1,2,3,4,5,6]

df_train['Farebin'] = pd.cut(df_train['Fare'], bins=bins, labels=labels)

df_test['Farebin']=pd.cut(df_test['Fare'],bins=bins,labels=labels)
df_train['Farebin'].unique()
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()
df_train['Sex']=labelencoder.fit_transform(df_train['Sex'])

df_test['Sex']=labelencoder.fit_transform(df_test['Sex'])
df_train.head()
df_train['Fam']=df_train['SibSp']+df_train['Parch']

df_test['Fam']=df_test['SibSp']+df_test['Parch']
df_train['AgebinnedI']=df_train['Agebinned'].astype(int)

df_test['AgebinnedI']=df_test['Agebinned'].astype(int)

df_train['FarebinI']=df_train['Farebin'].astype(int)

df_test['FarebinI']=df_test['Farebin'].astype(int)
df_train.dtypes
sns.countplot(df_train['Pclass'],hue=df_train['Sex'])
sns.countplot(df_train['Sex'])
sns.countplot(df_train['Agebinned'],hue=df_train['Sex']) #Check Age Bins For more Info
sns.countplot(df_train['Fam'],hue=df_train['Sex'])
sns.countplot(df_train['Farebin'])
sns.countplot(df_train['Embarked'])
features=['Pclass','Sex','Agebinned','Fam','Farebin','Embarked']

for i in features:

    plt.figure()

    sns.barplot(df_train[i],df_train['Survived'])
# df['lunch'] = (df['hour']<=11) & (df['hour']<=1)

df_train['IsAlone']=(df_train['Fam']==0)

df_train['IsAlone']=df_train['IsAlone'].astype(int)

df_test['IsAlone']=(df_test['Fam']==0)

df_test['IsAlone']=df_test['IsAlone'].astype(int)

df_train.head()
df_newtrain=df_train[['Sex','Q','S','p1','p2','Fam','AgebinnedI','FarebinI','IsAlone']]

df_newtest=df_test[['Sex','Q','S','p1','p2','Fam','AgebinnedI','FarebinI','IsAlone']]

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df_newtrain.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
x_train = df_newtrain.copy()

y_train = df_train["Survived"]

x_test  = df_newtest.copy()

x_train.shape, y_train.shape, x_test.shape
################################################LOGISTIC REGRESSION#######################################################

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(df_newtrain.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
##############################################Support Vector Machines#######################################3



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
###################################K-Nearest-Neighbours###############################

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

yt_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

acc_knn
###########################################Gaussian Naive Bayes############################################

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

acc_gaussian
#############################################Perceptron#################################################

perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_test)

acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

acc_perceptron
#####################################Linear SVC#####################################################

linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

acc_linear_svc
#################################Stochastic Gradient Descent#########################################################

sgd = SGDClassifier()

sgd.fit(x_train,y_train)

y_pred = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train,y_train) * 100, 2)

acc_sgd
##########################################Decision Tree#########################################################

decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
################################################Random Forest######################################################

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

Ytx_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

acc_random_forest
###############################################XGBOOST#####################################################

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.5, gamma=0, subsample=1,

                           colsample_bytree=1, max_depth=10)

xgb.fit(x_train,y_train)

Yt_pred= xgb.predict(x_test)

xgb.score(x_train, y_train)

acc_xgb = round(xgb.score(x_train, y_train) * 100, 2)

acc_xgb
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree','XGBoost'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree,acc_xgb]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Ytx_pred

    })
submission.to_csv('submission.csv', index=False)