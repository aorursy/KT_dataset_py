import pandas as pd

import numpy as np

import random as rnd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVR



from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
train_df.columns.values
train_df.describe()
train_df.head()
train_df.info()

print("__"*30)

test_df.info()
train_df[["Survived","Pclass"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df[["Survived","Sex"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df[["Survived","SibSp"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df[["Survived","Parch"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', height=3.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by="Survived",ascending=False)
to_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(to_map)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.sample(10)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
combine
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for dataset in combine:

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    dataset["Age"]=dataset["Age"].astype(int)

train_df.sample(20)

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].sample(10)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(train_df['Embarked'].mode()[0])

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.sample(10)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.sample(10)
test_df
#X_train = train_df.drop("Survived", axis=1)

#Y_train = train_df["Survived"]

#X_test  = test_df.drop("PassengerId", axis=1).copy()

X=train_df.iloc[:,1:]

y=train_df.iloc[:,0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43421)
classification_models = [KNeighborsClassifier(n_neighbors = 5),

                        SVC(probability=True),

                        DecisionTreeClassifier(),

                        RandomForestClassifier(n_estimators=100),

                        AdaBoostClassifier(),

                        GradientBoostingClassifier(),

                        GaussianNB(),

                        LinearDiscriminantAnalysis(),

                        QuadraticDiscriminantAnalysis(),

                        LogisticRegression(),

                        XGBClassifier()]

names=[]

acc=[]

pre=[]

con=[]
for clf in classification_models:

        name = clf.__class__.__name__

        names.append(name)

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        #accuracy = round(clf.score(y_test, train_predictions) * 100, 2)

        accuracy=accuracy_score(y_test, train_predictions)*100

        acc.append(accuracy)

        precision=precision_score(y_test, train_predictions)*100

        pre.append(precision)

        confusion=confusion_matrix(y_test, train_predictions)

        con.append(confusion)

        #pre=precision_score(X_train,Y_train)
model_dict = {"Models":names,

             "Confusion Matrix":con,

             "Accuracy":acc,

             "Precision":pre,

            #"Recall":Re,

            #"F1 score":F1

             }
model_df = pd.DataFrame(model_dict)

model_df
test_df
X_train
submission=test_df.iloc[:,1:]

regr=SVC()

regr.fit(X_train, y_train)

train_predictions = regr.predict(submission)

train_predictions

pd.DataFrame({"PassengerId": test_df["PassengerId"].values, "Survived": train_predictions.astype(int)}

).to_csv("submission.csv", index=False)