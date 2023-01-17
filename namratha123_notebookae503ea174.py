# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
combine=[train,test]
train.info()
train.describe()
train.describe(include=['O'])
pd.crosstab(train.Sex,train.Survived)
train['Sex'].unique()
pd.crosstab(train.Pclass,train.Survived)
pd.crosstab(train.Embarked,train.Survived)
import seaborn as sns
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age',bins=20)
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

pd.crosstab(train.Title,train.Survived)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
train = train.drop(['Name', 'PassengerId'], axis=1)

test = test.drop(['Name'], axis=1)

combine = [train, test]

train.shape, test.shape
train.Title.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train.Sex)
le.classes_
train.Sex=le.transform(train.Sex)
sns.lmplot(x = 'Pclass', y = 'Age', hue='Survived',

           col="Survived", data=train, col_wrap=3, size=3)
train.Age.hist()



df.fillna(df.median(),inplace=True)
df.isnull().sum()
df.dtypes
df['Cabin'].fillna("unknown",inplace=True)
df.isnull().sum()
df.dtypes


df['Embarked']


df['Embarked'].fillna("S",inplace=True)
df.isnull().sum()
df
df['survived_acc_sex']= 1
X=df[["Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
y=df["survived_acc_sex"]
df.dtypes
df['Sex'].unique()
df['Cabin'].unique()
df['Embarked'].unique()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(df['Sex'].unique())
df['Sex'] = le.transform(df['Sex'])
df['Sex'].unique()
le.fit(df['Embarked'].unique())


df['Embarked'] = le.transform(df['Embarked'])
df['Embarked'].unique()
df.dtypes
X=df[["Pclass","Sex","Age","Fare"]]
for i in df["Sex"]:

    if(i==0):

        df["survived_acc_sex"]=1

    else:

        df["survived_acc_sex"]=0
y=df["survived_acc_sex"]
y=df["Survived"]
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X)
accuracy_pred = accuracy_score(y_pred,y)
accuracy_pred
from sklearn.model_selection import StratifiedKFold
kf= StratifiedKFold(n_splits=5, random_state=3)
kf.split(X,y)
def kFold(X,y,clf):

    accuracy_from_dt = []

    for train_index,test_index in kf.split(X,y):

        train_X,test_X = X.iloc[train_index],X.iloc[test_index]

        train_y,test_y = y[train_index],y[test_index]

        clf.fit(train_X,train_y)

        prediction = clf.predict(test_X)

        

        #print(accuracy_score(prediction,test_y))

        accuracy_from_dt.append(accuracy_score(prediction,test_y))

        

    #print()

    #print("mean cv score")

    return np.mean(accuracy_from_dt)
kFold(X,y,clf)