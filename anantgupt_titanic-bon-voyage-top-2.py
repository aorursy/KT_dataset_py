import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns
# training data

train_dts = pd.read_csv('../input/titanic/train.csv')

train_dts.head()
# test data

test_dts = pd.read_csv('../input/titanic/test.csv')

test_dts.head()
female = train_dts.loc[train_dts.Sex=='female']['Survived']

print('% of Female survived : {:.3f}'.format((sum(female)/len(female))*100))



male = train_dts.loc[train_dts.Sex=='male']['Survived']

print('% of Male survived : {:.3f}'.format((sum(male)/len(male))*100))
print('Shape of Training Set : {}'.format(train_dts.shape))

print('Number of training data points : {}\n'.format(len(train_dts)))

print('Shape of Test Set : {}'.format(test_dts.shape))

print('Number of test data points : {}\n'.format(len(test_dts)))

print('Columns : {}'.format(train_dts.columns))

train_dts.info()
test_dts.info()
train_dts.describe()
test_dts.describe()
g = sns.heatmap(train_dts.corr(),annot=True, fmt = ".1f", cmap = "coolwarm")
age_hist = train_dts.Age.hist()
train_dts.groupby('Pclass').Survived.mean()
pd.crosstab(index=train_dts['Sex'], columns=train_dts['Pclass'], values=train_dts.Survived, aggfunc='mean')
embarked_hist = train_dts.Embarked.hist()
#Fill nan values in Embarked with 'S' as it is most frequent value

train_dts['Embarked'] = train_dts['Embarked'].fillna('S')

train_dts['Age'] = train_dts['Age'].fillna(train_dts['Age'].mean())

train_dts['Age'].isnull().sum() 
test_dts['Fare'] = test_dts['Fare'].fillna(test_dts['Fare'].median())

test_dts['Age'] = test_dts['Age'].fillna(test_dts['Age'].mean())
title = [i.split(",")[1].split(".")[0].strip() for i in train_dts['Name']]

train_dts['Title'] = pd.Series(title)



title_ = [i.split(",")[1].split(".")[0].strip() for i in test_dts['Name']]

test_dts['Title'] = pd.Series(title_)



train_dts.Title.value_counts()
train_dts["Title"] = train_dts["Title"].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don',  'Dr',

                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle'],

                                                'Rare'

                                               )

test_dts["Title"] = test_dts["Title"].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr',

                                               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle'],

                                              'Rare'

                                             )
plt.figure(figsize=(6,6))

plt.hist(train_dts.Title)

plt.xticks(rotation=45)

plt.show
mr = train_dts.loc[train_dts['Title']=='Mr'].Survived

miss = train_dts.loc[train_dts['Title']=='Miss'].Survived

mrs = train_dts.loc[train_dts['Title']=='Mrs'].Survived

master = train_dts.loc[train_dts['Title']=='Master'].Survived

rare = train_dts.loc[train_dts['Title']=='Rare'].Survived



print("probablity of Surviving if Mr : {:.2f}".format(sum(mr)/len(mr)))

print("probablity of Surviving if Mrs : {:.2f}".format(sum(mrs)/len(mrs)))

print("probablity of Surviving if Miss : {:.2f}".format(sum(miss)/len(miss)))

print("probablity of Surviving if Master : {:.2f}".format(sum(master)/len(master)))

print("probablity of Surviving if Rare : {:.2f}".format(sum(rare)/len(rare)))
g = sns.catplot(x="Title", y="Survived", data=train_dts, kind='bar').set_ylabels("Survival Probability")
train_dts['FamilySize'] = train_dts['SibSp'] + train_dts['Parch'] + 1

test_dts['FamilySize'] = test_dts['SibSp'] + test_dts['Parch'] + 1
g = sns.catplot(data=train_dts, x='FamilySize', y='Survived', kind='point').set_ylabels("Survival Probability")
# on training set

train_dts['Singleton'] = train_dts['FamilySize'].map(lambda s: 1 if s == 1 else 0)

train_dts['SmallFamily'] = train_dts['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

train_dts['LargeFamily'] = train_dts['FamilySize'].map(lambda s: 1 if 5 <= s else 0)



# on test set

test_dts['Singleton'] = test_dts['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test_dts['SmallFamily'] = test_dts['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

test_dts['LargeFamily'] = test_dts['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
train_dts.loc[train_dts.Cabin.isnull(), 'Cabin'] = 0

train_dts.loc[train_dts.Cabin != 0, 'Cabin'] = 1



test_dts.loc[test_dts.Cabin.isnull(), 'Cabin'] = 0

test_dts.loc[test_dts.Cabin != 0, 'Cabin'] = 1



train_dts['Cabin'] = pd.to_numeric(train_dts['Cabin'])

test_dts['Cabin'] = pd.to_numeric(test_dts['Cabin'])
train_dts.Ticket.describe()
def cleanTicket(ticket):

    ticket = ticket.replace('.','')

    ticket = ticket.replace('/','')

    ticket = ticket.split()

    ticket = ticket[0]

    if ticket.isdigit():

        return 'X'

    else:

        return ticket[0]

    

train_dts['Ticket'] = train_dts['Ticket'].map(cleanTicket)

test_dts['Ticket'] = test_dts['Ticket'].map(cleanTicket)
train_dts.Ticket.unique()
train_dts.min()
train_dts.max()
X_train = pd.DataFrame.copy(train_dts)

X_test = pd.DataFrame.copy(test_dts)



# label encoding

X_train = pd.get_dummies(X_train, columns=['Sex', 'Embarked', 'Title', "Pclass", 'Ticket'])

X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked', 'Title', "Pclass", 'Ticket'])

X_train.shape
# droping columns

X_train.drop(labels=['Name', 'PassengerId', 'Survived'], axis=1, inplace=True)

X_test.drop(labels=['Name', 'PassengerId'], axis=1, inplace=True)
plt.figure(figsize = (14,14))

g = sns.heatmap(X_train.corr(),annot=True, fmt = ".1f", cmap = "coolwarm")
y_train = train_dts.Survived
X_train.info()

X_train.head()
X_test.info()

X_test.head()
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators= 2000,

                        max_depth= 4,

                        min_child_weight= 2,

                        gamma=0.9,                    

                        subsample=0.8,

                        colsample_bytree=0.8,

                        objective= 'binary:logistic',

                        nthread= -1,

                        scale_pos_weight=1

                       )



xgb_clf.fit(X_train, y_train)



# testing on train set

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(xgb_clf.predict(X_train), y_train))

print('Accuracy of training')

print(accuracy_score(xgb_clf.predict(X_train), y_train))
pred = pd.Series(xgb_clf.predict(X_test), name='Survived')

results = pd.concat([test_dts['PassengerId'], pred], axis=1)

results.to_csv("submission.csv", index=False)

results.head(10)