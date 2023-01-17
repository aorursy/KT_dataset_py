# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing, svm

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

%matplotlib inline



d_train = pd.read_csv('../input/train.csv')

d_test = pd.read_csv('../input/test.csv')



def modified_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    grps = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult',

              'Senior']

    categories = pd.cut(df.Age, bins, labels=grps)

    df.Age = categories

    return df



def modified_cabins(df):

    df.Cabin = df.Cabin.fillna('NA')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def modified_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    grps = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=grps)

    df.Fare = categories

    return df



def modified_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_feat(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = modified_ages(df)

    df = modified_cabins(df)

    df = modified_fares(df)

    df = modified_name(df)

    df = drop_feat(df)

    return df



d_train = transform_features(d_train)

d_test = transform_features(d_test)

#d_train.head()



def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

d_train, d_test = encode_features(d_train, d_test)

#d_train.head()



X_all = d_train.drop(['Survived', 'PassengerId'], axis=1)

y_all = d_train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)



clf = LogisticRegression()





parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



acc_scorer = make_scorer(accuracy_score)



#grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

#grid_obj = grid_obj.fit(X_train, y_train)





#clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))



ids = d_test['PassengerId']

predictions = clf.predict(d_test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('final-predictions_LR.csv', index = False)

#output.tail()


