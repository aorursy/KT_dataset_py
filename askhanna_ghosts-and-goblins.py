# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('../input/test.csv')
df = pd.read_csv('../input/train.csv')
df.head()
df.drop(axis=1, columns=['id'],inplace=True)
df.head()
df.describe()
df['type'].value_counts()
sns.pairplot(data=df,hue='type')
df.columns
fig,ax = plt.subplots(figsize = (16,10), ncols=2,nrows=2)
sns.boxplot(data=df,x='type', y='bone_length', ax=ax[0][0])
sns.boxplot(data=df,x='type', y='rotting_flesh', ax=ax[0][1])
sns.boxplot(data=df,x='type', y='hair_length', ax=ax[1][0])
sns.boxplot(data=df,x='type', y='has_soul', ax=ax[1][1])
df['color'].value_counts()
df.groupby('color')['type'].value_counts()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

df['type'] = le.fit_transform(df['type'])
#df_test['type'] = le.transform(df_test['type'])

df_transformed = pd.get_dummies(df,drop_first=True,prefix=['color'])
df_transformed.columns
y = df_transformed['type']
X = df_transformed.loc[:,['bone_length', 'rotting_flesh', 'hair_length', 'has_soul','color_blood', 'color_blue', 'color_clear', 'color_green', 'color_white']]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, accuracy_score,make_scorer


X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=0)
accuracy_scorer = make_scorer(accuracy_score)
params = {'C': [1,2,3,5],'penalty':['l1','l2']}
logi = LogisticRegression(random_state=0)
clf_logi = GridSearchCV(estimator=logi,param_grid=params,scoring = accuracy_scorer,cv=5,n_jobs=-1)
clf_logi.fit(X_train,y_train)

print('Best score: {}'.format(clf_logi.best_score_))
print('Best parameters: {}'.format(clf_logi.best_params_))

#print(classification_report(y_test, y_pred))
#print("\nAccuracy Score is: " + str(accuracy_score(y_test, y_pred)))
y_true, y_pred = y_test, clf_logi.predict(X_test)
print(classification_report(y_true, y_pred))
params = {'n_estimators':[10,20,50,100], 'criterion': ['gini','entropy'], 'max_depth':[None, 5, 10, 25, 50]}
rf = RandomForestClassifier(random_state = 0)
clf_rf = GridSearchCV(estimator=rf, param_grid=params,scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf_rf.fit(X_train,y_train)
print('Best score: {}'.format(clf_rf.best_score_))
print('Best parameters: {}'.format(clf_rf.best_params_))

#rf_best = RandomForestClassifier(n_estimators = 10, random_state = 0)
y_true, y_pred = y_test, clf_rf.predict(X_test)
print(classification_report(y_true, y_pred))
params = {'n_estimators':[10, 25, 50, 100], 'max_samples':[1, 3, 5, 10]}
bag = BaggingClassifier(random_state = 0)
clf_bag = GridSearchCV(bag, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf_bag.fit(X_train, y_train)
print('Best score: {}'.format(clf_bag.best_score_))
print('Best parameters: {}'.format(clf_bag.best_params_))

y_true, y_pred = y_test, clf_bag.predict(X_test)
print(classification_report(y_true, y_pred))
#bag_best = BaggingClassifier(max_samples = 5, n_estimators = 25, random_state = 0)
params = {'kernel': ['rbf','linear'], 'C': [1,3,5,10],'degree': [3,5,10],'gamma':[0.001,0.01,0.1,1,10]}
svc = SVC(probability = True, random_state = 0)
clf_svc = GridSearchCV(estimator=svc, param_grid=params,scoring=accuracy_scorer,cv=5,n_jobs=-1)
clf_svc.fit(X_train,y_train)
print('Best score: {}'.format(clf_svc.best_score_))
print('Best parameters: {}'.format(clf_svc.best_params_))

y_true, y_pred = y_test, clf_svc.predict(X_test)
print(classification_report(y_true, y_pred))

params = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5,6,7,8,9,10]}

dectree = DecisionTreeClassifier()
clf_dec = GridSearchCV(estimator=dectree,param_grid=params,scoring=accuracy_scorer,cv=5,n_jobs=-1)
clf_dec.fit(X_train,y_train)

print('Best score: {}'.format(clf_dec.best_score_))
print('Best parameters: {}'.format(clf_dec.best_params_))

y_true, y_pred = y_test, clf_dec.predict(X_test)
print(classification_report(y_true, y_pred))