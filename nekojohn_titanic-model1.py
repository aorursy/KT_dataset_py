# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

pd.options.display.max_columns = None

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
print(train.head(10))
test = pd.read_csv('../input/test.csv')
print(test.head(10))
print(train.shape)
print(test.shape)
print(train.isnull().sum())
print(test.isnull().sum())
print(train.info())
print(train.describe(include=['O']))
print(train.describe(include='all'))
print(train.describe())
import seaborn as sns
sns.set(style="ticks")
sel_columns = ['Survived','Pclass']
sns.pairplot(train[sel_columns], hue="Survived")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
# check missing age vs non-missing age
import numpy as np
train['mis_age'] = np.where(pd.isna(train['Age'])==True,'missing','not_missing')
pct_mis_age = train[['mis_age','Survived']].groupby(['mis_age'],as_index=False).mean()

sns.barplot(x='mis_age',y='Survived', data=pct_mis_age)
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'SibSp', bins=20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'SibSp',shade= True)
facet.set(xlim=(0, train['SibSp'].max()))
facet.add_legend()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Parch',shade= True)
facet.set(xlim=(0, train['Parch'].max()))
facet.add_legend()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
train['family'] = train.Parch + train.SibSp
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'family',shade= True)
facet.set(xlim=(0, train['family'].max()))
facet.add_legend()
# Sex
pct_sex = train[['Survived','Sex']].groupby(['Sex'],as_index=False).mean()
sns.barplot(x='Sex', y='Survived', data=pct_sex, order=['male','female'])
sns.factorplot('Sex','Survived', data=train)
# Pclass
pct_class = train[['Survived','Pclass']].groupby(['Pclass'],as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', data=pct_class)
# Embarked
pct_embarked = train[['Survived','Embarked']].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=pct_embarked)
train.Cabin.value_counts()
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
cat_features = ['Pclass','Sex','Embarked']
X_dummy = pd.get_dummies(train[cat_features])
# drop Embarked_S and Sex_male
#X_dummy.drop(['Embarked_S','Sex_male'],inplace=True)
X_dummy.drop(['Sex_male','Embarked_S'],axis = 1, inplace=True)
print(X_dummy.head(10))
# cat variables: Pclass, Sex, Embarked
# numeric : family, Age, Fare


features = ['family','Age','Fare','Pclass', 'Sex', 'Embarked']
X = pd.get_dummies(train[features], columns=['Sex','Embarked'])
X.drop(['Sex_male','Embarked_S'],axis = 1, inplace=True)
print(X.head(10))

seed = 7
test_size = 0.2
Y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



model = XGBClassifier(min_child_weight=10, n_estimators=200)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('accuracy =', accuracy)
print('population bad rate=',np.mean(y_test))
features = ['family','Age','Fare','Pclass', 'Sex', 'Embarked']
test['family'] = test.Parch + test.SibSp
test_x = pd.get_dummies(test[features], columns=['Sex','Embarked'])
test_x.drop(['Sex_male','Embarked_S'],axis = 1, inplace=True)
pred_test = model.predict(test_x)
print(pred_test[:10])
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred_test
    })
submission.to_csv('titanic.csv', index=False)
model1 = XGBClassifier(learning_rate=0.01, n_estimators=2000)
model1.fit(X_train, y_train)
pred_test1 = model.predict(test_x)
print(pred_test1[:10])
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred_test
    })
submission.to_csv('titanic_high_learn_rate.csv', index=False)
