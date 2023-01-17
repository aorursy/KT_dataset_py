# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
datasets = pd.read_csv('../input/titanic.csv')
datasets.describe()
datasets.info()
datasets.head()
datasets.tail()
datasets.shape
f,ax = plt.subplots(figsize=(10, 5))
sns.heatmap(datasets.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
datasets.drop(columns = ['PassengerId','Name','Cabin','Parch','Ticket','Embarked'], inplace = True)
datasets.head()
datasets.rename(columns = {'Survived' : 'Hayatta','Pclass' : 'Sinif','Age' : 'Yas'},inplace = True)
datasets.rename(columns = {'Sex' : 'Cinsiyet','SibSp' : 'Aile','Fare' : 'Ucret'},inplace = True)
datasets.head()
f,ax = plt.subplots(figsize=(10, 5))
sns.heatmap(datasets.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
datasets.Ucret.plot(kind = 'line', color = 'b',label = 'Ucret',linewidth=0.5,alpha = 1,grid = True,linestyle = ':')
datasets.Hayatta.plot(color = 'r',label = 'Hayatta',linewidth=2, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Ucret')              # label = name of label
plt.ylabel('Hayatta')
plt.title('Line Plot')            # title = title of plot
plt.show()
datasets.Yas.plot(kind = 'hist',bins = 50,figsize = (8,5))
plt.show()
datasets.isnull().sum()
medyan = datasets.Yas.median()
medyan
datasets['Yas'] = datasets.Yas.fillna(medyan)
datasets.head(15)
datasets.isnull().sum().sum()
dummies = pd.get_dummies(datasets.Cinsiyet)
datasets = pd.concat([datasets,dummies],axis = 1)
datasets = datasets.drop(['Cinsiyet','female'],axis = 1)
datasets.head(15)
X = datasets.drop(columns = ['Hayatta'])
y = datasets.Hayatta.values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
X_train.shape
X_test.shape
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 44, max_depth=3)
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
print("Accuracy of Random Forest Classifier Algorithm:",random_forest.score(X_test,y_test))
print("Random Forest Classifier roc_auc skoru : {}".format(roc_auc_score(y_test,y_pred)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
f,ax = plt.subplots(figsize=(8, 4))
sns.heatmap(cm, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
from sklearn.svm import SVC
svm = SVC(random_state = 44)
svm.fit(X_train,y_train)
y_head = svm.predict(X_test)
print("Accuracy of Support Vector Machine Algorithm : ",svm.score(X_test,y_test))
print("Support Vector Machine roc_auc skoru : {}".format(roc_auc_score(y_test,y_head)))
cm2 = confusion_matrix(y_head,y_test)
f,ax = plt.subplots(figsize=(8, 4))
sns.heatmap(cm2, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
y_guess = nb.predict(X_test)
cm3 = confusion_matrix(y_guess,y_test)
f,ax = plt.subplots(figsize=(8, 4))
sns.heatmap(cm3, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
print("Naive Bayes Classifier Algorithm : ",nb.score(X_test,y_test))
print("Naive Bayes Classifier roc_auc skoru : {}".format(roc_auc_score(y_test,y_guess)))
