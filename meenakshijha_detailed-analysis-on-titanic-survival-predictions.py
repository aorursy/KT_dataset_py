# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women=train_data.loc[train_data.Sex=='female']['Survived']

rate_women=sum(women)/len(women)

print('{} proportion of women Survived on titanic. '.format(rate_women))
men=train_data.loc[train_data.Sex=='male']['Survived']

rate_men=sum(men)/len(women)

print('{} proportion of men Survived on titanic. '.format(rate_men))
plt.subplots(figsize=(5,5))

a=train_data.groupby(["Survived"])['Sex'].count()

colors = ['pink', 'yellow']



a.plot.pie(colors=colors,autopct='%1.1f%%',shadow=True, startangle=175)
plt.subplots(figsize=(5,5))

a=train_data.groupby(["Sex"]).Sex.count()

colors = ['red', 'blue']



a.plot.pie(colors=colors,autopct='%1.1f%%',shadow=True, startangle=175)
pd.crosstab(train_data.Sex,train_data.Survived,margins=True).style.background_gradient(cmap='summer_r')

train_data.shape
columns=train_data.columns

for i in columns:

    print(i,'has :',train_data[i].nunique(),'Unique Values')
train_data.isnull().sum()
train_data.info()
plt.figure(figsize=(8,5))

sns.countplot('Survived',data=train_data,palette="rainbow")

plt.title('Survived')

plt.show()

print(train_data[['Survived']].count())

train_data['Survived'].value_counts()
print('The % of people reported dead  on the Titanic:',(train_data['Survived'].value_counts()[0]/np.sum(train_data['Survived'].value_counts()))*100)

print('The % of people who survived on the Titanic:',(train_data['Survived'].value_counts()[1]/np.sum(train_data['Survived'].value_counts()))*100)
sns.set(style='darkgrid')

fig,axes=plt.subplots(1,2,figsize=(18,8))

train_data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=axes[0])

axes[0].set_title('Number Of Passengers By Pclass')

axes[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=train_data,ax=axes[1])

axes[1].set_title('Pclass:Survived vs Dead')

plt.show()

print(train_data[['Pclass']].count())

print(train_data['Pclass'].value_counts())

plt.figure(figsize=(10,7))

sns.countplot(train_data.Pclass,hue=train_data.Sex,palette="plasma_r")

pd.crosstab([train_data.Sex,train_data.Survived],train_data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
sns.set(style='whitegrid')

f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train_data,palette="OrRd_r",ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train_data,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()

train_data['Age'].mean()
train_data['Prefix']=0

for i in train_data:

    train_data['Prefix']=train_data.Name.str.extract('([A-Za-z]+)\.')
train_data['Prefix'].unique()
train_data['Prefix'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train_data.groupby(['Prefix'])['Age'].mean()
train_data.loc[(train_data.Age.isnull())&(train_data.Prefix=='Mr'),'Age']=33

train_data.loc[(train_data.Age.isnull())&(train_data.Prefix=='Mrs'),'Age']=36

train_data.loc[(train_data.Age.isnull())&(train_data.Prefix=='Master'),'Age']=5

train_data.loc[(train_data.Age.isnull())&(train_data.Prefix=='Miss'),'Age']=22

train_data.loc[(train_data.Age.isnull())&(train_data.Prefix=='Other'),'Age']=46
train_data['Age'].isnull().any()
sns.set(style="darkgrid")

fig,ax=plt.subplots(1,2,figsize=(15,8))

train_data[train_data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('DEATHS')

train_data[train_data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')

ax[1].set_title('SURVIVALS')

plt.show()
plt.figure(figsize=(12,5))

sns.set(style="darkgrid")

sns.countplot(train_data.Embarked,hue=train_data.Survived,palette='gist_heat_r')

plt.figure(figsize=(10,7))

sns.countplot(y="Pclass", hue="Sex", data=train_data,palette="cool_r")
fig,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=train_data,ax=ax[0,0],palette="CMRmap")

ax[0,0].set_title('Port Of Embarkment')

sns.countplot('Embarked',hue='Pclass',data=train_data,ax=ax[0,1],palette="autumn_r")

ax[0,1].set_title('Embarked port V/S Pclass')

sns.countplot('Embarked',hue='Survived',data=train_data,ax=ax[1,0],palette="CMRmap_r")

ax[1,0].set_title('Embarked port V/S Survived')

sns.countplot('Embarked',hue='Sex',data=train_data,ax=ax[1,1],palette="icefire_r")

ax[1,1].set_title('Embarked port V/S Sex')

plt.show()

train_data['Embarked'].fillna('S',inplace=True)
train_data.Embarked.isnull().any()
fig,ax=plt.subplots(1,2,figsize=(15,8))

sns.barplot('SibSp','Survived',data=train_data,ax=ax[0])

ax[0].set_title('Sibsp V/S Survived')

sns.barplot('SibSp','Pclass',hue='Survived',data=train_data,ax=ax[1])

ax[1].set_title('Sibsp in Different Pclass V/S Survived')

plt.show()
pd.crosstab([train_data.Pclass,train_data.Survived],train_data.SibSp).style.background_gradient(cmap='summer_r')
train_data.isnull().sum()
train_data['Cabin'].values
drop_column = ['PassengerId','Cabin', 'Ticket']

train_data.drop(drop_column, axis=1, inplace = True)
train_data.head()
pd.crosstab(train_data.Parch,train_data.Survived).style.background_gradient(cmap='summer_r')
train_data['Parch'].value_counts()
train_data.Parch.count()
train_data['Parch'].unique()
plt.figure(figsize=(12,10))

corr_back = train_data.corr()





mask = np.zeros_like(corr_back, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(corr_back, mask=mask, center=0, square=True, linewidths=.5)



plt.show()
train_data.info()
train_data['Sex'] = pd.get_dummies(data=train_data['Sex'],drop_first=True)
train_data['Embarked'] = pd.get_dummies(data=train_data['Embarked'],drop_first=True)
train_data['Name'] = pd.get_dummies(data=train_data['Name'],drop_first=True)
train_data['Prefix'] = pd.get_dummies(data=train_data['Prefix'],drop_first=True)
X = train_data[['Pclass', 'Fare', 'Parch','Sex', 'Embarked']]

y = train_data['Survived']
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
dt=DecisionTreeClassifier(max_depth=4)

dt.fit(X_train,y_train)

d_pred=lr.predict(X_test)

print(accuracy_score(y_test,d_pred))
# Create the parameter grid 

param_grid = {

    'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]

}



n_folds = 5



# Instantiate the grid search model

dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 

                          cv = n_folds, verbose = 1)



# Fit the grid search to the data

grid_search.fit(X_train,y_train)
grid_search.best_params_
grid_search.best_score_
dt_default1 = DecisionTreeClassifier(max_depth=5,min_samples_leaf=100,min_samples_split=50,criterion='entropy')

dt_default1.fit(X_train, y_train)

dt_pred=dt_default1.predict(X_test)

print(accuracy_score(y_test,dt_pred))
import xgboost as XG
xg = XG.XGBClassifier(objective='binary:logistic',

n_estimators=10, seed=101)

xg.fit(X_train, y_train)

preds = xg.predict(X_test)

accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))

test_data['Sex'] = pd.get_dummies(data=test_data['Sex'],drop_first=True)
test_data['Embarked']=pd.get_dummies(data=test_data['Embarked'],drop_first=True)
test_data.drop(['Name','Ticket','Cabin','Age'],axis=1,inplace=True)

test_data.drop(['SibSp'],axis=1,inplace=True)
test_data['Fare'] = test_data['Fare'].fillna(0)
test_data['Fare'] = test_data['Fare'].astype('int')
X_test_set = test_data[['Pclass', 'Fare','Parch','Sex', 'Embarked']]
predictions = xg.predict(X_test_set)
submission = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})

submission.Survived = submission.Survived.astype(int)

print(submission.shape)

filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

print("Your submission was successfully saved!")