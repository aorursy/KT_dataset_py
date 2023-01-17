import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from fancyimpute import  KNN

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')

import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
tot= pd.concat([train.drop('Survived',1),test])
survived = train['Survived']
y=train['Survived']
X=train.iloc[:, 1:10]
X.head()
pd.crosstab(train.Sex, train.Survived) 
tot.head(3)
tot['Sex'] = tot['Sex'].map({'male': 0, 'female': 1})
tot.drop(['Ticket','Cabin', 'Embarked'], axis=1,inplace=True)
tot.describe()
tot_num = tot.select_dtypes(include=[np.number])
tot_numeric = tot.select_dtypes(include=[np.number]).as_matrix()
tot_filled = pd.DataFrame(KNN(3).complete(tot_numeric))
tot_filled.columns = tot_num.columns
tot_filled.index = tot_num.index
print(tot_filled.info(), tot_filled.describe())
tot=tot_filled
test = tot.iloc[len(train):]
train = tot.iloc[:len(train)]
train['Survived'] = survived
surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
y_train=train['Survived']
X_train=train.iloc[:, 1:7]
X_tree = X_train[['Sex', 'Age']]
plt.figure(figsize=[15,10])
sns.distplot(train['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue', axlabel='Age')
msurv = train[(train['Survived']==1) & (train['Sex']==0)]
fsurv = train[(train['Survived']==1) & (train['Sex']==1)]
mnosurv = train[(train['Survived']==0) & (train['Sex']==0)]
fnosurv = train[(train['Survived']==0) & (train['Sex']==1)]
plt.figure(figsize=[15,10])
sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue', axlabel='Age')
sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red')
plt.figure(figsize=[15,10])
plt.subplot(211)
sns.distplot(fsurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(fnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red',
            axlabel='Female Age')
plt.subplot(212)
sns.distplot(msurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(mnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red',
            axlabel='Male Age')
from sklearn import tree
model = tree.DecisionTreeClassifier(min_samples_split=10, max_depth=3, min_samples_leaf=50)
model
model.fit(X_tree, y_train)
y_predict = model.predict(X_tree)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_predict)
import graphviz
dot_data = tree.export_graphviz(model, feature_names=X_tree.columns, out_file=None, filled=True, rounded=True, special_characters=True)
graphviz.Source(dot_data)
plt.figure(figsize=[15,10])
plt.subplot(211)
plt.axvline(x=24, color='black')
plt.axvline(x=31, color='black')
plt.axvline(x=41, color='black')
sns.distplot(fsurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(fnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red',axlabel='Female Age')
plt.subplot(212)
plt.axvline(x=16, color='black')
plt.axvline(x=25, color='black')
sns.distplot(msurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
sns.distplot(mnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Male Age')