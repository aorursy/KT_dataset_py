from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

import numpy as np

import pandas as pd

import os

import re



### define function ###

def print_description(_df):

    print('----- Data Description -----\n', _df.describe(include='all'))

    print('----- Sum of NULL -----\n', _df.isnull().sum())

    print('----- Sample -----\n', _df.sample(10))



def extract_features(_df):

    # create family size feature

    _df['FamilySize'] = _df['SibSp'] + _df['Parch'] + 1

    

    # create is_alone feature

    _df['IsAlone'] = 0

    _df['IsAlone'] = _df['IsAlone'].mask(_df['FamilySize'] == 1, 1)

    

    # create title of name feature

    title_list = ['Mr', 'Miss', 'Mrs', 'Master']

    _df['Title'] = _df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    _df['Title'] = _df['Title'].apply(lambda x: x if x in title_list else 'ETC')

    

    # process ticket feature

    _df['Ticket'] = _df['Ticket'].apply(

        lambda x: int(re.findall('\d+', str(x))[0]) 

        if any(i.isdigit() for i in str(x)) else np.nan)

    

    return _df



def clean_data(_df):

    _df['Age'].fillna(_df['Age'].median(), inplace=True)

    _df['Embarked'].fillna(_df['Embarked'].mode()[0], inplace=True)

    _df['Fare'].fillna(_df['Fare'].median(), inplace=True)

    _df['Ticket'].fillna(_df['Ticket'].median(), inplace=True)

    

    drop_columns = ['Cabin', 'Name']

    _df.drop(drop_columns, axis=1, inplace=True)

    

    return _df





### main logic ###

# load dataset

tr = pd.read_csv('../input/train.csv')  # train data frame

ts = pd.read_csv('../input/test.csv')  # test data frame



# analyze data

#print_description(tr)

#print_description(ts)



# preprocess (extracting features, cleaning and converting data)

tr = pd.get_dummies(clean_data(extract_features(tr)))

ts = pd.get_dummies(clean_data(extract_features(ts)))



X = tr.iloc[:, 2:].values

y = tr.iloc[:, 1].values



# train model

estimators = [('rf1', RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=300)),

            ('rf2', RandomForestClassifier(criterion='gini', max_depth=7, n_estimators=1100)),

            ('rf3', RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=350,

                                          min_samples_leaf=4, max_leaf_nodes=10, min_impurity_decrease=0, max_features=1)),

            ('rf4', RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=1000)),

            ('svm1', SVC(kernel='rbf', C=0.473, gamma=0.0736, probability=True)),

            ('svm2', SVC(kernel='rbf', C=0.5, gamma=0.5, probability=True)),

            ('svm3', SVC(kernel='rbf', C=0.7, gamma=0.7, probability=True)),

            ('svm4', SVC(kernel='rbf', C=0.1, gamma=0.1, probability=True)),

            ('svm5', SVC(kernel='rbf', C=2, gamma=0.1, probability=True)),

           ]



ensemble = VotingClassifier(estimators=estimators, voting='soft')

pipe = make_pipeline(StandardScaler(), ensemble)

pipe.fit(X, y)



# check model performance using cross validation

scores = cross_val_score(estimator=pipe, X=X, y=y, cv=10, n_jobs=-1)

print('Accuracy: %.3f +- %.3f' % (np.mean(scores), np.std(scores)))



# save result

X_test = ts.iloc[:, 1:].values

y_test_pred = pipe.predict(X_test)



passenger = ts.iloc[:, 0]

label = pd.DataFrame(y_test_pred)

result = pd.concat([passenger, label], axis=1)

result.columns = ['PassengerId', 'Survived']



result.to_csv('submission.csv', index=False)