### data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

passenger_id = test.PassengerId
train_df['Name_Len'] = train_df['Name'].apply(lambda x: len(x))

train_df['Survived'].groupby(pd.qcut(train_df['Name_Len'],5)).mean()
def names(train, test):

    for i in [train, test]:

        i['Name_Len'] = i['Name'].apply(lambda x: len(x))

        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

        del i['Name']

    return train, test





def age_impute(train, test):

    for i in [train, test]:

        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

        data = train.groupby(['Name_Title', 'Pclass'])['Age']

        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))

    return train, test





def fam_size(train, test):

    for i in [train, test]:

        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',

                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))

        del i['SibSp']

        del i['Parch']

    return train, test





def ticket_grouped(train, test):

    for i in [train, test]:

        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])

        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))

        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],

                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                            'Low_ticket', 'Other_ticket'))

        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))

        del i['Ticket']

    return train, test





def cabin(train, test):

    for i in [train, test]:

        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])

        del i['Cabin']

    return train, test





def cabin_num(train, test):

    for i in [train, test]:

        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])

        i['Cabin_num1'].replace('an', np.NaN, inplace = True)

        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)

    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)

    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)

    del train['Cabin_num']

    del test['Cabin_num']

    del train['Cabin_num1']

    del test['Cabin_num1']

    return train, test





def embarked_impute(train, test):

    for i in [train, test]:

        i['Embarked'] = i['Embarked'].fillna('S')

    return train, test    
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):

    for column in columns:

        train[column] = train[column].apply(lambda x: str(x))

        test[column] = test[column].apply(lambda x: str(x))

        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]

        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)

        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)

        del train[column]

        del test[column]

    return train, test
def drop(train, test, bye = ['PassengerId']):

    for i in [train, test]:

        for z in bye:

            del i[z]

    return train, test


train, test = names(train, test)

train, test = age_impute(train, test)

train, test = cabin_num(train, test)

train, test = cabin(train, test)

train, test = embarked_impute(train, test)

train, test = fam_size(train, test)

test['Fare'].fillna(train['Fare'].mean(), inplace = True)

train, test = ticket_grouped(train, test)

train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',

                                              'Cabin_Letter', 'Name_Title', 'Fam_Size'])

train, test = drop(train, test)
print(len(train.columns))
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)



param_grid = { "criterion" : ["entropy"], 

              "min_samples_leaf" : [6, 4, 2], 

              "min_samples_split" : [12], 

              "n_estimators": [825, 800, 775, 750]}



gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)



gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])



print(gs.best_params_)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(criterion='entropy', 

                             n_estimators=775,

                             min_samples_split=12,

                             min_samples_leaf=2,

                             max_features='auto',

                             oob_score=True,

                             random_state=2,

                             n_jobs=-1)

rf.fit(train.iloc[:, 1:], train.iloc[:, 0])

Y_pred = rf.predict(test)

print("%.4f" % rf.oob_score_)
output = pd.DataFrame({'PassengerId': passenger_id, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)