import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn import svm

import matplotlib.pyplot as plt





train_df = pd.read_csv('../input/train.csv')

real_test_df = pd.read_csv('../input/test.csv')



# all_data now has the training set + test set. The keys argument makes our new all_data

# a hierarchical index

# all_data.ix['train']

df = pd.concat([train_df, real_test_df], keys=['train', 'test'])



df
import re



df['title'] = df['Name'].str.extract(' (?P<title>[a-zA-Z]+)\. ', expand=True)

print(" The titles found in the names are %s" % df['title'].unique())
cabine_data = df['Ticket'].str.extract('^(?P<TicketLetter>[a-zA-Z0-9\\\/.]* )?(?P<TickerNumber>[0-9]+)$', expand=False)

df['TicketLetter'] = cabine_data['TicketLetter']

df['TickerNumber'] = cabine_data['TickerNumber']



# Lots of cabineletter that I think mean the same. Eg. A./.5 vs A/5 vs A/5.

# Best way I think is ignore dots, spaces and slashes

for char in '.\\/ ':

    df['TicketLetter'] = df['TicketLetter'].str.replace(char, '')



print(" The TicketLetter found in the names are %s \n\n" % df['TicketLetter'].unique())



print("We have %s items with a cabine letter" % df['TicketLetter'].count())



survived_by_cabineletter = df.ix['train'][['PassengerId', 'TicketLetter', 'Survived']].groupby(['TicketLetter', 'Survived'])

survived_by_cabineletter.size().unstack().plot.bar(stacked=True)



survived_by_cabinenumber = df.ix['train'][['PassengerId', 'TickerNumber', 'Survived']].groupby(['TickerNumber', 'Survived'])

survived_by_cabinenumber.size().unstack().plot.line(stacked=True)



# looks like the ticket number is not relevant
df['Age'].fillna(df['Age'].median(), inplace=True)



df
df_without_columns
# I've migrated Str data to small subster but we need to categorize them with floats.

# Pandas recommended way is to use get_dummies

cols_to_transform = [ 'Sex', 'title', 'TicketLetter' ]

df = pd.get_dummies(df, columns = cols_to_transform )





columns_we_dont_want_in_training = ['Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId', 'Ticket', 'CabinLetter', 'CabinNumber', 'TickerNumber']



df_without_columns = df[[col for col in list(df) if col not in columns_we_dont_want_in_training]]



df_without_columns
from sklearn.linear_model import LogisticRegression



X_train, X_test, y_train, y_test = train_test_split(df_without_columns.ix['train'], df_without_columns.ix['train']['Survived'], test_size=0.33, random_state=42)



# instantiate a logistic regression model, and fit with X and y

model = LogisticRegression()



del X_train['Survived']

del X_test['Survived']



model.fit(X_train, y_train)

model.score(X_test, y_test)
y_predicted = model.predict(X_test)
X_test['Survived'] = y_test

X_test['Predicted'] = y_predicted



# 50 mislabeled

X_test.loc[X_test['Survived'] != X_test['Predicted']][['Age', 'Pclass', 'SibSp', 'Sex_female', 'Sex_male', 'Predicted', 'Survived']]
names = list(X_train.columns.values)
from sklearn.feature_selection import RFE, RFECV



if 'Survived' in X_test:

    del X_test['Survived']

if 'Predicted' in X_test:

    del X_test['Predicted']



# not sure what the cv fold is

rfe = RFECV(model, cv=6)

rfe.fit(X_train, y_train)

print("Score is %s" % rfe.score(X_test, y_test))

print("Features sorted by their rank:")

print(rfe.ranking_)

print(rfe.support_)

print("Optimal number of features : %d (original number %s)" % (rfe.n_features_, len(names)))
X_submit = df_without_columns.ix['test']



if 'Survived' in X_submit:

    del X_submit['Survived']

    

y_predicted = model.predict(X_submit)



df.ix['test']['PassengerId']

import os



X_submit['Survived'] = y_predicted



X_submit['PassengerId'] = df.ix['test']['PassengerId']



to_submit = X_submit[['PassengerId', 'Survived']]



# todo: find a better way

to_submit[['PassengerId', 'Survived']] = to_submit[['PassengerId', 'Survived']].astype(int)



# this score leads to 0.79426

# to_submit.to_csv('/tmp/titanic_logistic.csv', index = False)