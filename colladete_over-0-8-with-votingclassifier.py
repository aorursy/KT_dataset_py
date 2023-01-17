# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import re



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

import xgboost as xgb

from mlxtend.classifier import StackingClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit



import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.drop(['PassengerId'], 1, inplace=True)

test_df.drop(['PassengerId'], 1, inplace=True)
train_df.head()
ticketgrouptrain = train_df.groupby('Ticket').size().reset_index()

ticketgrouptest = test_df.groupby('Ticket').size().reset_index()

ticketgrouptrain.rename(columns = {0: 'Count'}, inplace=True)

ticketgrouptest.rename(columns = {0: 'Count'}, inplace=True)



for i in range(0,train_df.shape[0]):

    train_df.ix[i, 'Sharedticket'] = ticketgrouptrain.Count[ticketgrouptrain.Ticket == train_df.Ticket[i]].item()

for i in range(0,test_df.shape[0]):

    test_df.ix[i, 'Sharedticket'] = ticketgrouptest.Count[ticketgrouptest.Ticket == test_df.Ticket[i]].item()   



train_df.drop('Ticket', 1, inplace=True)

test_df.drop('Ticket', 1, inplace=True)   
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Sharedticket', y = 'Survived', data = train_df, ax = axs[0])

sns.barplot(x=train_df.groupby('Sharedticket').size().index, 

            y = train_df.groupby('Sharedticket').size().values, ax = axs[1])
train_df['Sharedticket2'] = train_df.Sharedticket

train_df.Sharedticket2[(train_df.Sharedticket2 == 2.0) | (train_df.Sharedticket2 == 3.0)  | 

                      (train_df.Sharedticket2 == 4.0) ] = 2

train_df.Sharedticket2[(train_df.Sharedticket2 == 5.0) | (train_df.Sharedticket2 == 6.0) | 

                       (train_df.Sharedticket2 == 7.0)] = 3
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Sharedticket2', y = 'Survived', data = train_df, ax = axs[0])

sns.barplot(x=train_df.groupby('Sharedticket2').size().index, y = train_df.groupby('Sharedticket2').size().values, ax = axs[1])
train_df.Sharedticket[train_df.Sharedticket == 1] = 'Not shared'

train_df.Sharedticket[(train_df.Sharedticket == 2.0) | (train_df.Sharedticket == 3.0)  | 

                      (train_df.Sharedticket == 4.0) ] = 'Shared small'

train_df.Sharedticket[(train_df.Sharedticket == 5.0) | (train_df.Sharedticket == 6.0) | 

                       (train_df.Sharedticket == 7.0)] = 'Shared big'

test_df.Sharedticket[test_df.Sharedticket == 1] = 'Not shared'

test_df.Sharedticket[(test_df.Sharedticket == 2.0) | (test_df.Sharedticket == 3.0)  | 

                      (test_df.Sharedticket == 4.0) ] = 'Shared small'

test_df.Sharedticket[(test_df.Sharedticket == 5.0) | (test_df.Sharedticket == 6.0) | 

                       (test_df.Sharedticket == 7.0)] = 'Shared big'

train_df.drop('Sharedticket2', 1, inplace=True)
train_df['Family'] = train_df['SibSp'] + train_df['Parch']

test_df['Family'] = test_df['SibSp'] + test_df['Parch']
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Family', y = 'Survived', data = train_df, ax = axs[0])

sns.countplot(train_df['Family'], ax = axs[1])
for i in range(0, train_df.shape[0]):

    if train_df.Family[i] == 0:

        train_df.ix[i, 'Familytype'] = 'Alone'

    elif (train_df.Family[i] > 0) & (train_df.Family[i] < 4):

        train_df.ix[i, 'Familytype'] = 'Small family'

    else:

        train_df.ix[i, 'Familytype'] = 'Big family'



for i in range(0, test_df.shape[0]):

    if test_df.Family[i] == 0:

        test_df.ix[i, 'Familytype'] = 'Alone'

    elif (test_df.Family[i] > 0) & (test_df.Family[i] < 4):

        test_df.ix[i, 'Familytype'] = 'Small family'

    else:

        test_df.ix[i, 'Familytype'] = 'Big family'   

        

train_df.drop(['SibSp', 'Parch', 'Family'], 1, inplace=True)

test_df.drop(['SibSp', 'Parch', 'Family'], 1, inplace=True) 
tab = pd.crosstab(train_df['Sharedticket'], train_df['Familytype'])

print(tab)
train_df[train_df.Embarked.isnull()]
sns.barplot(x='Embarked', y ='Fare', data= train_df)
train_df.Embarked[train_df.Embarked.isnull()] = 'C'
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Embarked', y = 'Survived', data = train_df, ax = axs[0])

sns.countplot(train_df['Embarked'], ax = axs[1])
train_df.Cabin[train_df.Cabin.isnull()] = 'Z'

train_df.Cabin = [x[0] for x in train_df.Cabin]

test_df.Cabin[test_df.Cabin.isnull()] = 'Z'

test_df.Cabin = [x[0] for x in test_df.Cabin]
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Cabin', y = 'Survived', data = train_df, ax = axs[0])

sns.countplot(train_df['Cabin'], ax = axs[1])
train_df.ix[(train_df.Cabin != 'Z'), 'Cabinknown'] = 'Yes'

train_df.ix[(train_df.Cabin == 'Z'), 'Cabinknown'] = 'No'

test_df.ix[(test_df.Cabin != 'Z'), 'Cabinknown'] = 'Yes'

test_df.ix[(test_df.Cabin == 'Z'), 'Cabinknown'] = 'No'

train_df.drop('Cabin', 1, inplace=True)

test_df.drop('Cabin', 1, inplace=True)
sns.barplot(x='Cabinknown', y = 'Survived', data = train_df)
test_df[test_df.Fare.isnull()]

train_df.Fare[(train_df['Pclass'] == 3) & (train_df['Embarked'] == 'S')].mean()

test_df.Fare[test_df.Fare.isnull()] = 14.64
sns.boxplot(x=train_df.Survived, y= np.log10(train_df.Fare + 1))
train_df.ix[(train_df.Fare >= 70),'Fare2'] = 3

train_df.ix[(train_df.Fare < 70),'Fare2'] = 2

train_df.ix[(train_df.Fare <= 10),'Fare2'] = 1

test_df.ix[(test_df.Fare >= 70),'Fare2'] = 3

test_df.ix[(test_df.Fare < 70),'Fare2'] = 2

test_df.ix[(test_df.Fare <= 10),'Fare2'] = 1
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Fare2', y = 'Survived', data = train_df, ax = axs[0])

sns.countplot(x='Fare2', data = train_df, ax = axs[1])
train_df.drop('Fare', 1, inplace=True)

test_df.drop('Fare', 1, inplace=True)
for i in range(0,train_df.shape[0]):

    train_df.ix[i, 'Title'] = re.search('([A-Za-z]+)\.', train_df.Name[i]).group(0)

for i in range(0,test_df.shape[0]):

    test_df.ix[i, 'Title'] = re.search('([A-Za-z]+)\.', test_df.Name[i]).group(0)
train_df.Title.value_counts()
test_df.Title.value_counts()
train_df.Title[(train_df.Title == 'Dr.') | (train_df.Title == 'Rev.') | (train_df.Title == 'Major.') | 

               (train_df.Title == 'Col.') | (train_df.Title == 'Sir.') | (train_df.Title == 'Don.') |

              (train_df.Title == 'Capt.') | (train_df.Title == 'Jonkheer.')] = 'Mr.'

train_df.Title[(train_df.Title == 'Mlle.') | (train_df.Title == 'Ms.') | (train_df.Title == 'Lady.')] = 'Miss.'

train_df.Title[(train_df.Title == 'Mme.') | (train_df.Title == 'Countess.')] = 'Mrs.'

test_df.Title[(test_df.Title == 'Col.') | (test_df.Title == 'Rev.') | (test_df.Title == 'Dr.')] = 'Mr.'

test_df.Title[(test_df.Title == 'Dona.')] = 'Mrs.'

test_df.Title[(test_df.Title == 'Ms.')] = 'Miss.'
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.barplot(x='Title', y = 'Survived', data = train_df, ax = axs[0])

sns.countplot(train_df['Title'], ax = axs[1])
fig, axs = plt.subplots(ncols=2, figsize=(12,4))

sns.distplot(train_df.Age[-train_df.Age.isnull() & (train_df.Survived == 1) & (train_df.Sex == 'male')], kde = False, ax = axs[0])

sns.distplot(train_df.Age[-train_df.Age.isnull() & (train_df.Survived == 0) & (train_df.Sex == 'male')], color = 'red', kde = False, ax = axs[0])

sns.distplot(train_df.Age[-train_df.Age.isnull() & (train_df.Survived == 1) & (train_df.Sex == 'female')], kde = False, ax = axs[1])

sns.distplot(train_df.Age[-train_df.Age.isnull() & (train_df.Survived == 0) & (train_df.Sex == 'female')], color = 'red', kde = False, ax = axs[1])
train_df.Title[(train_df.Title == 'Miss.') | (train_df.Title == 'Mrs.')] = 'Woman'

test_df.Title[(test_df.Title == 'Miss.') | (test_df.Title == 'Mrs.')] = 'Woman'

train_df.rename(columns = {'Title': 'Person'}, inplace=True)

test_df.rename(columns = {'Title': 'Person'}, inplace=True)

train_df.drop(['Name', 'Sex'], 1, inplace=True)

test_df.drop(['Name', 'Sex'], 1, inplace=True)
train_df.ix[-np.isnan(train_df.Age), 'Ageknown'] = 'Yes'

train_df.ix[np.isnan(train_df.Age), 'Ageknown'] = 'No'

test_df.ix[-np.isnan(test_df.Age), 'Ageknown'] = 'Yes'

test_df.ix[np.isnan(test_df.Age), 'Ageknown'] = 'No'
sns.barplot(x='Ageknown', y = 'Survived', data = train_df)
sns.barplot(x = 'Pclass', y = 'Survived', data = train_df)
col_names = ['Cabinknown', 'Embarked', 'Person', 'Ageknown', 'Sharedticket', 'Familytype']

train_df = pd.get_dummies(train_df, columns=col_names, drop_first=True)

test_df = pd.get_dummies(test_df, columns=col_names, drop_first=True)

train_df.drop('Age', 1, inplace=True)

test_df.drop('Age', 1, inplace=True)
train_df.head()
X = np.array(train_df.drop('Survived', 1))

y = np.array(train_df['Survived'])

testX = np.array(test_df)
svcmod = SVC(C = 15, kernel = 'rbf', gamma = 0.2, probability= True)

svcmod.scores = cross_val_score(svcmod, X, y, cv=5)



knnmod = KNeighborsClassifier(n_neighbors= 21)

knnmod.scores = cross_val_score(knnmod, X, y, cv=5)



rfmod = RandomForestClassifier(n_estimators=100, max_features=10, max_depth=5)

rfmod.scores = cross_val_score(rfmod, X, y, cv=5)



xgbmod = xgb.XGBClassifier(learning_rate= 0.1, gamma = 0.8, max_depth= 3, min_child_weight=7, 

                           objective= 'binary:logistic', subsample=0.8, colsample_bytree=0.8, 

                           scale_pos_weight=1, n_estimators = 1000, nthread = -1)

xgbmod.scores = cross_val_score(xgbmod, X, y, cv=5)



etmod = ExtraTreesClassifier(n_estimators=300, max_features=6, max_depth=6)

etmod.scores = cross_val_score(etmod, X, y, cv=5)



admod = AdaBoostClassifier(n_estimators= 100, learning_rate= 0.1)

admod.scores = cross_val_score(admod, X, y, cv=5)



lr = LogisticRegression(C = 0.1)

lr.scores = cross_val_score(lr, X, y, cv=5)



for clf, label in zip([svcmod, knnmod, rfmod, xgbmod, etmod, admod, lr], 

                      ['SVC', 

                       'KNN', 

                       'Random Forest',

                       'XGB',

                       'ExtraTrees',

                       'AdaBoost',

                       'Logistic Regression']):

    print("%s Accuracy: %0.3f" % (label,clf.scores.mean()))
votemod = VotingClassifier(estimators=[('xgb', xgbmod), ('et', etmod), ('knn', knnmod), 

                                       ('rf', rfmod), ('ad', admod)], 

                           weights = [2,1,2,1,1],

                           voting='soft')

votemod.scores = cross_val_score(votemod, X, y, cv=5)

print(votemod.scores.mean())
stcmod = StackingClassifier(classifiers=[svcmod, xgbmod, knnmod, etmod, admod, rfmod], 

                            meta_classifier=lr)

stcmod.scores = cross_val_score(stcmod, X, y, cv=5)

print(stcmod.scores.mean())
votemod.fit(X,y)

predictionsfinal = votemod.predict(testX)



testfinal = pd.read_csv('../input/test.csv')



submit = pd.DataFrame({'PassengerId' : testfinal.loc[:,'PassengerId'],

                       'Survived': predictionsfinal})

submit.to_csv("../working/submit.csv", index=False)