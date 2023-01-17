# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold



sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
def bar_chart(features):

    survived = train[train['Survived']==1][features].value_counts()

    dead = train[train['Survived']==0][features].value_counts()

    df = pd.DataFrame([survived, dead])

    df.index = ['Survived', 'Dead']

    df.plot(kind='bar', stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
train_test_data = [train,test]



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract('([A-za-a]+)\.',expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, "Mile":3, "Jonkheer":3, "Countess":3, "Sir":3, "Ms":3, "Lady":3, "Mme":3, "Don":3, "Capt":3, "Dona":3}
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train.head()
train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
sex_mapping = {'male':0, 'female':1}



for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train.head()
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.head(20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

 

plt.show()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(0,20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(20,40)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(40,60)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(60,100)
for dataset in train_test_data:

    dataset.loc[dataset['Age']<=16, 'Age'] = 0

    dataset.loc[(dataset['Age']>16)& (dataset['Age']<=23), 'Age'] =1

    dataset.loc[(dataset['Age']>23)& (dataset['Age']<=34), 'Age'] =2

    dataset.loc[(dataset['Age']>34)& (dataset['Age']<=42), 'Age'] =3

    dataset.loc[(dataset['Age']>42)& (dataset['Age']<=58), 'Age'] =4

    dataset.loc[dataset['Age']>58, 'Age'] = 5
train.head(10)
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class', '2nd class', '3rd class']

df.plot(kind='bar', stacked=True, figsize=(10,5))
for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Title'] = dataset['Title'].fillna(0)
train.isnull().sum()
test.isnull().sum()
embarked_mapping = {'S': 0, "C": 1, "Q": 2}



for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

 

plt.show()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(0,20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(20,40)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(40,60)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(60,100)
for dataset in train_test_data:

    dataset.loc[dataset['Fare']<=17, 'Fare'] = 0

    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=29), 'Fare']= 1

    dataset.loc[(dataset['Fare']>29) & (dataset['Fare']<=100), 'Fare']= 2

    dataset.loc[dataset['Fare']>100, 'Fare'] = 3
train.head(10)
train.Cabin.value_counts()
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class', '2nd class', '3rd class']

df.plot(kind ='bar', stacked=True, figsize=(10,5))
cabin_mapping = {"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6, "F":2.0, "G":2.4, "T":2.8}

for dataset in train_test_data:

    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)
train['Cabin'].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test['Cabin'].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train.head(20)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['FamilySize'].max()
family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4.0}

for dataset in train_test_data:

    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)
train.head(10)
features_drop = ['SibSp', 'Ticket', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)
train_data = train.drop(['Survived'], axis=1)

target = train['Survived']
train_data.head(10)
train.info()
test.info()
k_fold = StratifiedKFold(n_splits=10)
random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, train_data, y = target, scoring = "accuracy", cv = k_fold, n_jobs=-1))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
DTC = DecisionTreeClassifier(max_depth=5)



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,10,20,30,40,50],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose = 1,return_train_score = True)



gsadaDTC.fit(train_data, target)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [5],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[500,1000],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsExtC.fit(train_data,target)



ExtC_best = gsExtC.best_estimator_
gsExtC.best_score_
RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [2],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,500],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=k_fold, scoring="accuracy", n_jobs=-1, verbose = 1)



gsRFC.fit(train_data,target)



RFC_best = gsRFC.best_estimator_

gsRFC.best_score_
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [1000,2000,3000],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [19],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(train_data,target)



GBC_best = gsGBC.best_estimator_
gsGBC.best_score_
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=k_fold, scoring="accuracy", n_jobs=-1, verbose = 1)



gsSVMC.fit(train_data,target)



SVMC_best = gsSVMC.best_estimator_
gsSVMC.best_score_
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=-1)



votingC = votingC.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1).copy()

prediction = votingC.predict(test_data)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })

submission.to_csv('submission.csv', index=False)