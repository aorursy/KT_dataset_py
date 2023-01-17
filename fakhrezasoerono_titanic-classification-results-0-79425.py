import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/titanic/train.csv')

df_new = pd.read_csv('../input/titanic/test.csv')

df = df.drop(columns='PassengerId')

df_new = df_new.drop(columns='PassengerId')
print(df.info())

print(df_new.info())
# Extract titles from 'Name'

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df_new['Title'] = df_new.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



mapping = {'Mr' : 'Mr', 'Dr' : 'Dr', 'Miss' : 'Ms', 'Ms' : 'Ms', 'Mrs' : 'Mrs', 'Master' : 'Master', 'Rev' : 'Rev', 'Miss' : 'Ms', 'Mlle' : 'Ms', 'Mme' : 'Mrs',

'Major' : 'Mill', 'Capt' : 'Mill', 'Col' : 'Mill', 'Lady' : 'Hon', 'Jonkheer' : 'Hon', 'Sir' : 'Hon', 'Don' : 'Hon', 'Countess' : 'Hon', 'Dona' : 'Hon'}



df.Title = df.Title.map(mapping)

df_new.Title = df_new.Title.map(mapping)

print('Title Extracted!')

df = df.drop(columns='Name')

df_new = df_new.drop(columns='Name')

print('Name Dropped!')
# Alone or Not

def alone_func(family):

    alone = []

    for x in family:

        if x == 0:

            alone.append(int(1))

        else:

            alone.append(int(0))

    return(alone)
df['Alone'] = alone_func(df.SibSp + df.Parch)

df_new['Alone'] = alone_func(df_new.SibSp + df_new.Parch)

print('Alone created!')
for alone in sorted(df.Alone.unique()):

    print('Alone ', alone, '({} data)'.format(len(df.Survived[df.Alone == alone])))

    print(df.Survived[df.Alone == alone].value_counts(normalize=True).sort_index(), '\n')
df = df.drop(columns=['SibSp', 'Parch'])

df_new = df_new.drop(columns=['SibSp', 'Parch'])

print('SibSp and Parch dropped!')
import re

def ticket_func(ticket_data):

    """Split ticket to a dataframe with TicketCat and TicketNo"""

    ticket_data = ticket_data.str.upper() # uppercase all entries

    ticket_data = ticket_data.str.replace('STON', 'SOTON') # fix alleged typos

    ticket_data = ticket_data.str.split(n=1, expand=True)

    for i, entry in ticket_data.iterrows():

        if ticket_data[0][i].isdecimal():

            ticket_data[1][i] = ticket_data[0][i]

            ticket_data[0][i] = 'no'

        ticket_data[0][i] = ''.join(re.findall('[a-zA-Z0-9/]', ticket_data[0][i])) # match alphanumerical and slash (/)

    ticket_data.columns = ['TicketCat', 'TicketNo']

    ticket_data['TicketNo'] = ticket_data['TicketNo'].fillna(0)

    return(ticket_data)
df[['TicketCat', 'TicketNo']] = ticket_func(df.Ticket)

df_new[['TicketCat', 'TicketNo']] = ticket_func(df_new.Ticket)

print(df[['TicketCat', 'TicketNo']].head())

print(df_new[['TicketCat', 'TicketNo']].head())
# Check TicketCat in test set which are not available in training set

sorted([x for x in df_new.TicketCat.unique() if not x in df.TicketCat.unique()])
df = df.drop(columns=['TicketCat', 'TicketNo', 'Ticket'])

df_new = df_new.drop(columns=['TicketCat', 'TicketNo', 'Ticket'])

print('Ticket dropped!')
def cabin_func(cabin_data):

    cabin_data = cabin_data.fillna('N') # fill nan with N

    cabin_data = cabin_data.str[0]

    cabin_data = cabin_data.str.replace('N', 'NO DATA')

    return(cabin_data)
df.Cabin = cabin_func(df.Cabin)

df_new.Cabin = cabin_func(df_new.Cabin)
print(sorted(df.Cabin.unique()))

print(sorted(df_new.Cabin.unique()))
for cabin in sorted(df.Cabin.unique()):

    print('Cabin ', cabin, '({} data)'.format(len(df.Survived[df.Cabin == cabin])))

    print(df.Survived[df.Cabin == cabin].value_counts(normalize=True).sort_index(), '\n')
print('Missing data in Cabin = {:.2f}% of total training data'.format(len(df[df.Cabin == 'NO DATA'])/len(df)*100))
# Cabin

df = df.drop(columns=['Cabin'])

df_new = df_new.drop(columns=['Cabin'])

print('Cabin Dropped!')
df.Age = df.Age.fillna(df.Age.mean()) # impute before binning

df_new.Age = df_new.Age.fillna(df.Age.mean()) # impute before binning
# Categorize Age Data (Binning)

def bin_age(age_data):

    age_category = []

    for age in age_data:

        if age <= 16:

            age_category.append('A')

        elif (age > 16) & (age <= 32):

            age_category.append('B')

        elif (age > 32) & (age <= 48):

            age_category.append('C')

        elif (age > 48) & (age <= 64):

            age_category.append('D')

        else:

            age_category.append('E')

    return age_category
df['AgeCategory'] = bin_age(df.Age)

df_new['AgeCategory'] = bin_age(df_new.Age)

print('AgeCategory created!')
for cat in sorted(df.AgeCategory.unique()):

    print('Age Category ', cat, '({} data)'.format(len(df.Survived[df.AgeCategory == cat])))

    print(df.Survived[df.AgeCategory == cat].value_counts(normalize=True).sort_index(), '\n')
df = df.drop(columns='Age')

df_new = df_new.drop(columns='Age')

print('Age dropped!')
# Impute

df.Embarked = df.Embarked.fillna(df.Embarked.value_counts().index[0])

df_new.Fare = df_new.Fare.fillna(df_new.Fare.mean())

print(df.info())

print(df_new.info())
# Check Class

print(df.Survived.value_counts(normalize=True))

print(df.Survived.value_counts())

df.Survived.value_counts(normalize=True).plot(kind='bar')
X = df.drop(columns='Survived')

X_pred = df_new

y = df.Survived

print('Training features:')

print(X.shape)

print('\nTest features:')

print(X_pred.shape)

print('\nTarget:')

print(y.shape)
# One Hot Encoder

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)

X_categorical = X.select_dtypes(object)

X_categorical = encoder.fit_transform(X_categorical)

X_pred_categorical = X_pred.select_dtypes(object)

X_pred_categorical = encoder.transform(X_pred_categorical)



X_numerical = X.select_dtypes(exclude=object).to_numpy()

X_pred_numerical = X_pred.select_dtypes(exclude=object).to_numpy()



X = np.concatenate((X_categorical, X_numerical), axis=1)

X_pred = np.concatenate((X_pred_categorical, X_pred_numerical), axis=1)



print('Training features:')

print(X.shape)

print('\nTest features:')

print(X_pred.shape)

print('\nTarget:')

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('Validation set (30%) ready!')
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.8)

lr.fit(X_train, y_train)

print(classification_report(y_test, lr.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 280)

rf.fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators = 280)

gb.fit(X_train, y_train)

print(classification_report(y_test, gb.predict(X_test)))
gb.fit(X, y)

rf.fit(X, y)

lr.fit(X, y)

print('Training completed!')
proba_gb = gb.predict_proba(X_pred)

proba_rf = rf.predict_proba(X_pred)

proba_lr = lr.predict_proba(X_pred)

proba_avg = (proba_gb + proba_rf + proba_lr) / 3

print('First 5 class probabilities:')

print(proba_avg[:5])
y_pred = []

for proba in proba_avg:

    if proba[0] > 0.5:

        y_pred.append(int(0))

    else:

        y_pred.append(int(1))

print('Target class ratio:')

print(pd.Series(y_pred).value_counts(normalize=True))
submission = pd.read_csv('../input/titanic/gender_submission.csv')

print(submission.head())

submission.Survived = y_pred

submission.set_index('PassengerId', inplace=True)

print(submission.head())
submission.to_csv('submission.csv')

print('Submission saved successfully!')