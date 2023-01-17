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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
train=pd.read_csv('../input/titanic/train.csv')

train.head()
train.info()
train.describe()
# male and female  distrubition

sns.stripplot(x='Sex',y='Age', data=train,jitter=True)
plt.figure(figsize=(12,8))

sns.stripplot(x='Sex',y='Age', data=train,jitter=True, hue='Pclass',dodge=True)



plt.figure(figsize=(12,8))

sns.swarmplot(x='Pclass',y='Age', data=train, hue='Survived',dodge=True)

plt.title('Survival with repect to Age group and Class')

plt.figure(figsize=(12,8))

sns.swarmplot(x='Sex',y='Age', data=train, hue='Survived',dodge=True)

plt.title('Survival with repect to Age group and Sex')
sns.countplot(x='Sex',data=train,saturation=1,dodge=False)

plt.title(' No. of male and Female passengers')
sns.countplot(x='Sex',data=train,saturation=0.9,dodge=True,hue='Pclass')

plt.title('No. of passenger with respect to Sex and Class')
sns.countplot(x='Sex',data=train,saturation=0.9,dodge=True,hue='Survived')

plt.title('No. of survived passenger with respect to Sex ')

plt.show()

sns.countplot(x='Pclass',data=train,saturation=0.9,dodge=True,hue='Survived')

plt.title('No. of survived passenger with respect to CLass')

plt.show
test=pd.read_csv('../input/titanic/test.csv')

test.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
train.drop(['PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)

test.drop(['PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)
train.info()
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age_train (cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 38

        elif Pclass==2:

            return 29

        else:

            return 24

    

    else:

        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age_train , axis=1)
train.isnull().sum()
train['Embarked'].fillna(method='bfill',inplace=True)
train.isnull().sum()
test['Fare'].mean()
test['Fare'].replace(to_replace=np.nan, value=35.6, inplace=True)
sns.boxplot(x='Pclass',y='Age',data=test)
def impute_age_test (cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 43

        elif Pclass==2:

            return 26

        else:

            return 24

    

    else:

        return Age
test['Age']=test[['Age','Pclass']].apply(impute_age_test , axis=1)
test.info()
sex=pd.get_dummies(train['Sex'],drop_first=True)

Embarked =pd.get_dummies(train['Embarked'],drop_first=True)

df_clean=pd.concat([train,sex,Embarked],axis=1)

df_clean.head()
df_clean.drop(['Embarked','Sex'],axis=1,inplace=True)

df_clean.info()
sex=pd.get_dummies(test['Sex'],drop_first=True)

Embarked =pd.get_dummies(test['Embarked'],drop_first=True)

test_clean=pd.concat([test,sex,Embarked],axis=1)

test_clean.drop(['Embarked','Sex'],axis=1,inplace=True)

test_clean.info()
X=df_clean.iloc[:,1:]

X.head()

y=df_clean.iloc[:,0:1]

y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))
from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,grid_predictions))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier

model=BaggingClassifier(LogisticRegression(),n_estimators=100)

model.fit(X_train,y_train)
model.score(X_test,y_test)
en_bagging=model.predict(X_test)
print(classification_report(y_test,en_bagging))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100, criterion='entropy' ,max_depth=3)
rfc.fit(X_train,y_train)
en_rand=rfc.predict(X_test)

print(classification_report(y_test,en_rand))
rfc.score(X_test,y_test)
param_grid={'n_estimators':[10,100,200],

            'criterion':['entropy','gini'],

            'max_depth':[3,4,5,6]}
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(RandomForestClassifier(),param_grid, refit=True, verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100, criterion='entropy' ,max_depth=6)

rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)