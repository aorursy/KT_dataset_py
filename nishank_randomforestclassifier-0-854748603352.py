import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import os

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
train.describe()
train.info()
categorical = train.dtypes[train.dtypes == "object"].index

print(categorical)



train[categorical].describe()
train.isnull().sum()
train[["Pclass", "Survived"]].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train);
train['Initial']=0

for i in train:

    train['Initial']=train.Name.str.extract('([A-Za-z]+)\.')

    

test['Initial']=0

for i in test:

    test['Initial']=train.Name.str.extract('([A-Za-z]+)\.')
print (train['Initial'].unique())

print (test['Initial'].unique())
train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

test['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme'],['Mr', 'Mrs', 'Miss', 'Master','Mr','Other','Mr','Mrs'],inplace=True)



print (train['Initial'].unique())

print (test['Initial'].unique())
train.groupby('Initial')['Age'].mean()
train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age']=33

train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age']=36

train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age']=5

train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age']=22

train.loc[(train.Age.isnull())&(train.Initial=='Other'),'Age']=46



test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=33

test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=36

test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=5

test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22

test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age']=46
bins = (0, 5, 12, 18, 25, 35, 60, 120)

group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

categories = pd.cut(train['Age'], bins, labels=group_names)

train['Age'] = categories
bins = (0, 5, 12, 18, 25, 35, 60, 120)

group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

categories = pd.cut(test['Age'], bins, labels=group_names)

test['Age'] = categories
train['Cabin'] = train['Cabin'].fillna('N')

train['Cabin'] = train['Cabin'].apply(lambda x: x[0])



test['Cabin'] = test['Cabin'].fillna('N')

test['Cabin'] = test['Cabin'].apply(lambda x: x[0])
train['Fare'].sort_values().unique()
sns.distplot(train['Fare'],bins=50)
group_names = ['1Q', '2Q', '3Q', '4Q']

quartiles = pd.qcut(train['Fare'],4, labels=group_names)
train['Fare']=quartiles

group_names = ['1Q', '2Q', '3Q', '4Q']

quartiles_test = pd.qcut(test['Fare'],4, labels=group_names)

test['Fare']=quartiles
train=train.drop(['Ticket', 'Name'], axis=1)

test=test.drop(['Ticket', 'Name'], axis=1)
pd.crosstab(index=train["Embarked"], columns="count")
train["Embarked"]=train["Embarked"].fillna("S")
train.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=train);
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train);
sns.barplot(x="Fare", y="Survived", hue="Sex", data=train);
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
from sklearn import preprocessing

def encode_features(train, test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Initial', 'Embarked']

    df_concat = pd.concat([train[features], test[features]])

    

    for feature in features:

        lab = preprocessing.LabelEncoder()

        lab = lab.fit(df_concat[feature])

        train[feature] = lab.transform(train[feature])

        test[feature] = lab.transform(test[feature])

    return train, test

    

train, test = encode_features(train, test)

train.head()
from sklearn.model_selection import train_test_split



X_all = train.drop(['Survived', 'PassengerId'], axis=1)

y_all = train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=18)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier()



parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



acc_scorer = make_scorer(accuracy_score)



grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



clf = grid_obj.best_estimator_



clf.fit(X_train, y_train)
# from sklearn.svm import SVC

# from sklearn.metrics import make_scorer, accuracy_score

# from sklearn.model_selection import GridSearchCV



# clf = SVC()



# parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 

#               'gamma': [0.001, 0.01, 0.1, 1],

#               'kernel': ['linear', 'poly', 'rbf', 'sigmoid']

#              }



# acc_scorer = make_scorer(accuracy_score)



# grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

# grid_obj = grid_obj.fit(X_train, y_train)



# clf = grid_obj.best_estimator_



# clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.cross_validation import KFold



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)
ids = test['PassengerId']

predictions = clf.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

#output.to_csv('titanic-prediction.csv', index = False)

output.head()