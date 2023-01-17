import pandas as pd 

import numpy as np

from sklearn.ensemble import RandomForestClassifier
train= pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

train.head(3)
train= train.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)

test= test.replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
train.isnull().sum()
train["Age"].fillna(train.Age.mean(), inplace=True) 

train["Embarked"].fillna(train.Embarked.mean(), inplace=True)
combine1 = [train]



for train in combine1: 

        train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 

for train in combine1: 

        train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')

        train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')

        train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')

        del train['Name']

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 

for train in combine1: 

        train['Salutation'] = train['Salutation'].map(Salutation_mapping) 

        train['Salutation'] = train['Salutation'].fillna(0)
for train in combine1: 

        train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])

        train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 

        train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 

        train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x)) 

        del train['Ticket'] 

train['Ticket_Lett']=train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)
for train in combine1: 

    train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 

    train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 

    train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))

del train['Cabin'] 

train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)
train.head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

for train in combine1:

    train['IsAlone'] = 0

    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train_data = train.values

xs = train_data[:, 2:] # Pclass以降の変数

y  = train_data[:, 1]  # 正解データ
test.info()
test["Age"].fillna(train.Age.mean(), inplace=True)

test["Fare"].fillna(train.Fare.mean(), inplace=True)



combine = [test]

for test in combine:

    test['Salutation'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for test in combine:

    test['Salutation'] = test['Salutation'].replace(['Lady', 'Countess','Capt', 'Col',\

         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    test['Salutation'] = test['Salutation'].replace('Mlle', 'Miss')

    test['Salutation'] = test['Salutation'].replace('Ms', 'Miss')

    test['Salutation'] = test['Salutation'].replace('Mme', 'Mrs')

    del test['Name']

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for test in combine:

    test['Salutation'] = test['Salutation'].map(Salutation_mapping)

    test['Salutation'] = test['Salutation'].fillna(0)



for test in combine:

        test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])

        test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x))

        test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],

                                   np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                            '0', '0'))

        test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))

        del test['Ticket']

test['Ticket_Lett']=test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 



for test in combine:

        test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0])

        test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x))

        test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin(['T', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'],

                                   np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                            '0','0'))        

        del test['Cabin']

test['Cabin_Lett']=test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1).replace("G",1) 



test["FamilySize"] = train["SibSp"] + train["Parch"] + 1



for test in combine:

    test['IsAlone'] = 0

    test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

    

test_data = test.values

xs_test = test_data[:, 1:]
'''

from sklearn.ensemble import RandomForestClassifier



random_forest=RandomForestClassifier()

random_forest.fit(xs, y)

Y_pred = random_forest.predict(xs_test)



import csv

with open("predict_result_data.csv", "w") as f:

    writer = csv.writer(f, lineterminator='\n')

    writer.writerow(["PassengerId", "Survived"])

    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):

        writer.writerow([pid, survived])

'''
from sklearn.ensemble import RandomForestClassifier

from sklearn import grid_search

from sklearn.grid_search import GridSearchCV

'''



parameters = {

        'n_estimators'      : [10,25,50,75,100],

        'random_state'      : [0],

        'n_jobs'            : [4],

        'min_samples_split' : [5,10, 15, 20,25, 30],

        'max_depth'         : [5, 10, 15,20,25,30]

}



clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)

clf.fit(xs, y)

 

print(clf.best_estimator_)

'''
random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=25, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=15,

            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=4,

            oob_score=False, random_state=0, verbose=0, warm_start=False)

random_forest.fit(xs, y)

Y_pred = random_forest.predict(xs_test)



import csv

with open("predict_result_data.csv", "w") as f:

    writer = csv.writer(f, lineterminator='\n')

    writer.writerow(["PassengerId", "Survived"])

    for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):

        writer.writerow([pid, survived])
%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns

g = sns.factorplot(x="Sex", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
sns.countplot(x='Sex', data = train)
g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.factorplot(x="Salutation", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

del train['PassengerId']

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
sns.countplot(x='FamilySize', data = train, hue = 'Survived')
sns.countplot(x='FamilySize', data = train,hue = 'Pclass')
t=pd.read_csv("../input/train.csv").replace("S",0).replace("C",1).replace("Q",2)

train['Embarked']= t['Embarked']

g = sns.factorplot(x="Embarked", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
sns.countplot(x='Embarked', data = train,hue = 'Pclass')
sns.countplot(x='Embarked', data = train,hue = 'Sex')
plt.figure()

sns.FacetGrid(data=t, hue="Survived", aspect=4).map(sns.kdeplot, "Age", shade=True)

plt.ylabel('Passenger Density')

plt.title('KDE of Age against Survival')

plt.legend()
for t in combine1: 

    t.loc[ t['Age'] <= 15, 'Age']                                                = 0

    t.loc[(t['Age'] > 15) & (t['Age'] <= 25), 'Age'] = 1

    t.loc[(t['Age'] > 25) & (t['Age'] <= 48), 'Age'] = 2

    t.loc[(t['Age'] > 48) & (t['Age'] <= 64), 'Age'] = 3

    t.loc[ t['Age'] > 64, 'Age'] =4

g = sns.factorplot(x="Age", y="Survived",  data=t,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
sns.countplot(x='Age', data = t,hue = 'Sex')
sns.countplot(x='Age', data = t,hue = 'Survived')