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



train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
train.head(5)

train.info()
train.describe()
train=train.drop(['PassengerId'], axis=1)
sns.catplot(x="Pclass",col="Survived", data=train , kind="count",height=4, aspect=.7);

sns.catplot(x="Sex",col="Survived", data=train , kind="count",height=4, aspect=.7);

sns.catplot(x="SibSp",col="Survived", data=train , kind="count",height=4, aspect=.7);

sns.catplot(x="Parch",col="Survived", data=train , kind="count",height=4, aspect=.7);

sns.catplot(x="Embarked",col="Survived", data=train , kind="count",height=4, aspect=.7);

g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
train["Name"].head()
train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]

train["Title"] = pd.Series(train_title)

train["Title"].head()
g = sns.countplot(x="Title",data=train)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# Convert to categorical values Title 

train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":2, "Mr":3, "Rare":4})

train["Title"] = train["Title"].astype(int)

train.describe()
g = sns.countplot(train["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/","Mrs","Mr","Rare"])
# Create a family size descriptor from SibSp and Parch

train["Fsize"] = train["SibSp"] + train["Parch"] + 1
train.describe()
g = sns.factorplot(x="Fsize",y="Survived",data = train)

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

train['Single'] = train['Fsize'].map(lambda s: 1 if s == 1 else 0)

train['SmallF'] = train['Fsize'].map(lambda s: 1 if  s == 2  else 0)

train['MedF'] = train['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train['LargeF'] = train['Fsize'].map(lambda s: 1 if s >= 5 else 0)

train.describe()
g = sns.catplot(x="Single",y="Survived",data=train,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="SmallF",y="Survived",data=train,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="MedF",y="Survived",data=train,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="LargeF",y="Survived",data=train,kind="bar")

g = g.set_ylabels("Survival Probability")
train.head()
train["Title_orig"]=train["Title"]

train["Embarked_orig"]=train["Embarked"]

train.head()
# convert to indicator values Title and Embarked 

train = pd.get_dummies(train, columns = ["Title"])

train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")

train.describe()
train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin'] ])

train.describe()
g = sns.countplot(train["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.catplot(y="Survived",x="Cabin",data=train,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
train["Cabin_orig"]=train["Cabin"]

train.head()
train = pd.get_dummies(train, columns = ["Cabin"],prefix="Cabin")

train.describe()
Ticket = []

for i in list(train.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

train["Ticket"] = Ticket

train["Ticket"].head()
g = sns.countplot(train["Ticket"])
train.Age.isnull().sum()
train.groupby(['Pclass','Title_orig']).mean()['Age'].size
train["Age_orig"]=train["Age"]

train.head()
train["Age"] = train.groupby(['Pclass','Title_orig'])['Age'].transform(lambda x: x.fillna(x.mean()))
train.Age.isnull().sum()
for x in [1,2,3]:    ## for 3 classes

    train.Age[train.Pclass == x].plot(kind="kde")

plt.title("Age wrt Pclass")

plt.legend(("1st","2nd","3rd"))
for x in [1,2,3]:    ## for 3 classes

    train.Age_orig[train.Pclass == x].plot(kind="kde")

plt.title("Age_orig wrt Pclass")

plt.legend(("1st","2nd","3rd"))
#Import in what is needed (repeated np, pd and plt)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score

from sklearn import model_selection

from sklearn import preprocessing

from xgboost import XGBClassifier

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier



#Import data

#train = pd.read_csv('../input/train.csv')

#test = pd.read_csv('../input/test.csv')

combine = train.drop(["Survived","Name","Age_orig" ,"Cabin_orig", "Embarked_orig", "Sex"], axis=1).append(test).drop(["PassengerId","Cabin", "Ticket","Embarked","Name","Sex"], axis=1)

target = train['Survived']



#Defining learning vectors

nb = train.shape[0]

X = combine[:nb]

y = target



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, train_size=0.9)

combine.head(5)
#XGBoost model tuning

model = XGBClassifier(booster='gbtree', silent=1, seed=0, base_score=0.5, subsample=0.75)

parameters = {'n_estimators':[75], #50,100

            'max_depth':[4],#1,10

            'gamma':[4],#0,6

            'max_delta_step':[1],#0,2

            'min_child_weight':[1], #3,5 

            'colsample_bytree':[0.55,0.6,0.65], #0.5,

            'learning_rate': [0.001,0.01,0.1]

            }

tune_model =  GridSearchCV(model, parameters, cv=3, scoring='accuracy')

tune_model.fit(X_train,y_train)

print('Best parameters :', tune_model.best_params_)

print('Results :', format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))





#Learn on the whole data

tune_model.fit(X, y)

Y_pred = tune_model.predict(combine[nb:])



#Submit the prediction

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })
submission.to_csv('submission.csv', index=False)