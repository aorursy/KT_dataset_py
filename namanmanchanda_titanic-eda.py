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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
df_test.head()
print(df_train.shape)

print(df_test.shape)
# Correlation between different features and Survived

df_train.corr()
# Heatmap of the correlation

sns.heatmap(df_train.corr())

plt.show()
# Count of Survived people

sns.set_style('dark')

sns.set_palette('RdBu')

sns.set_context('poster')

sns.catplot(x = 'Survived',data=df_train, kind='count')

plt.show()
# Count of Survived people belonging to different Pclass

sns.set_palette(['Red','Green'])

sns.catplot(x = 'Pclass',data=df_train, kind='count',hue='Survived')

plt.show()
# Count of Survived people of each Sex

sns.set_palette(['Red','Green'])

sns.catplot(x = 'Sex', data = df_train, kind='count', hue='Survived')

plt.show()
# Distribution of Age among the Survived people

sns.set_context('notebook')

sns.catplot(x = 'Survived', y='Age', data=df_train,kind='box')

plt.show()
# Relation between the survived people and their fare

sns.catplot(x = 'Survived', y='Fare', data=df_train, kind='bar')

plt.show()
# Count of Survived people from each Embarking

sns.catplot(x = 'Embarked', data = df_train, kind='count', hue='Survived')

plt.show()
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_train['Cabin'] = df_train['Cabin'].fillna('Missing')

df_test['Cabin'] = df_test['Cabin'].fillna('Missing')

df_train = df_train.dropna()

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
df_train.isnull().sum()
df_test.isnull().sum()
df_train = df_train.drop(columns=['Name'],axis=1)

df_test = df_test.drop(columns=['Name'],axis=1)

df_train = df_train.drop(columns=['Ticket'],axis=1)

df_test = df_test.drop(columns=['Ticket'],axis=1)

df_train = df_train.drop(columns=['Cabin'], axis=1)

df_test = df_test.drop(columns=['Cabin'], axis=1)
sex_map = {

        'male':0,

    'female':1

}

df_train.loc[: ,'Sex'] = df_train['Sex'].map(sex_map)

df_test.loc[: , 'Sex'] = df_test['Sex'].map(sex_map)
df_train = pd.get_dummies(df_train, prefix_sep='_',columns=['Embarked'])

df_test = pd.get_dummies(df_test, prefix_sep='_',columns=['Embarked'])
df_train.head()
df_test.head()
print(df_train.shape)

print(df_test.shape)
# Base Models

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN



# Ensembling Techniques

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb



# Metrics 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
X = df_train.drop('Survived',axis=1)

y = df_train['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Base Classifiers

lr = LogisticRegression(max_iter=10000)

knn=KNN()

dt = DecisionTreeClassifier()

classifiers = [('LogisticRegression',lr),

              ('KNeighborsClassifier',knn),

              ('ClassificationTree',dt)]

for clf_name, clf in classifiers:

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf_name, 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
# Voting Classifier

vc = VotingClassifier(estimators = classifiers)

vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)

print('Voting Classifier', 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
# AdaBoost Classifier

adb_clf = AdaBoostClassifier(base_estimator = dt, n_estimators = 100)

adb_clf.fit(X_train, y_train)

y_pred = adb_clf.predict(X_test)

print('AdaBoostClassifier', 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
# GradientBoosting Classifier

gbt = GradientBoostingClassifier()

gbt.fit(X_train, y_train)

y_pred = gbt.predict(X_test)

print('GradientBoostingClassifier', 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
# Stochastic GradientBoostingClassifier

sgbt = GradientBoostingClassifier(max_depth=1,subsample=0.8,max_features=0.2,n_estimators=300,random_state=21)

sgbt.fit(X_train, y_train)

y_pred = sgbt.predict(X_test)

print('Stochastic GradientBoostingClassifier', 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
# XGBoost

xg_cl = xgb.XGBClassifier(objective='binary:logistic',

                         seed=123)

xg_cl.fit(X_train, y_train)

y_pred = xg_cl.predict(X_test)

print('XGBoost', 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
## The highest ROC AUC Score and Accuracy is given by GradientBoostingClassifier

# GradientBoostingClassifier

sgbt = GradientBoostingClassifier(max_depth=1,subsample=0.8,max_features=0.2,n_estimators=300,random_state=21)

sgbt.fit(X_train, y_train)

y_pred = sgbt.predict(X_test)

print('Stochastic GradientBoostingClassifier', 'Accuracy Score' , accuracy_score(y_test,y_pred) , " " , 'ROC AUC Score' , roc_auc_score(y_test, y_pred))
y_pred = sgbt.predict(df_test)
df_test['Survived'] = y_pred
df_test
df_submission = df_test.drop(["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"],axis=1)
df_submission.head()
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gender_submission.head()
df_submission.to_csv('results.csv',index=False)