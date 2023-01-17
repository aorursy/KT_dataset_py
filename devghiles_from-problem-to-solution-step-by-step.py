import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
# read all the data in the train file

data = pd.read_csv('../input/train.csv')
data.head()
data.count()
def preprocess(df):

    # fill in the missing values

    df['Age'].fillna(df['Age'].median(), inplace=True)

    df['Cabin'].fillna('UNK', inplace=True)

    df['Embarked'].fillna('UNK', inplace=True)

    

    # categorical data encoding

    enc = LabelEncoder()

    columns_to_encode = ['Sex', 'Ticket', 'Fare', 'Cabin', 'Embarked']

    for column in columns_to_encode:

        df[column] = enc.fit_transform(df[column])

    

    # drop the 'PassengerId' and 'Name' columns

    df.drop('PassengerId', axis=1, inplace=True)

    df.drop('Name', axis=1, inplace=True)
preprocess(data)

print(data.count())

data.head()
data.describe()
# empirical distributions of the features and the target

for column in data.columns:

    plt.hist(data[column], weights=np.ones(len(data)) / len(data))

    plt.title('{0} empirical distribution'.format(column))

    plt.xlabel('{0}'.format(column))

    plt.ylabel('Empirical PDF')

    plt.show()
# correlation

corr = data.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
plt.hist(data['Age'].loc[data['Survived'] == 1], color='g', label='Survived')

plt.hist(data['Age'].loc[data['Survived'] == 0], color='r', label='Died')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.legend()
X = data.drop('Survived', axis=1).values

y = data['Survived'].values

clf = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
encoders = dict()
def feature_extraction(df, test=False):

    # Age, SibSp and Parch

    X = np.concatenate((df['Age'].values.reshape(-1, 1),

                       df['SibSp'].values.reshape(-1, 1),

                       df['Parch'].values.reshape(-1, 1)),

                       axis=1)

    

    # one-hot encoding of Pclass, Sex and Embarked

    for column in ['Pclass', 'Sex', 'Embarked']:

        if test:

            values = np.array(encoders[column].transform(df[column].values.reshape(-1, 1)).todense())

        else:

            encoders[column] = OneHotEncoder(handle_unknown='ignore')

            values = np.array(encoders[column].fit_transform(df[column].values.reshape(-1, 1)).todense())

        X = np.concatenate((X, values), axis=1)

    

    # For columns Ticket, Fare and Cabin, we only keep the most common values

    num_values_to_keep = {

        'Ticket': 8,

        'Fare': 15,

        'Cabin': 4

    }

    for column in ['Ticket', 'Fare', 'Cabin']:

        if not test:

            counts = Counter(df[column])

            most_common_counts = counts.most_common(num_values_to_keep[column])

            values_to_keep = list(map(lambda x: x[0], most_common_counts))

            encoders[column] = OneHotEncoder(handle_unknown='ignore')

            encoders[column].fit(np.array(values_to_keep).reshape(-1, 1))

        values = np.array(encoders[column].transform(df[column].values.reshape(-1, 1)).todense())

        X = np.concatenate((X, values), axis=1)

    

    return X
X = feature_extraction(data)

y = data['Survived'].values
clfs = {

    'mnb': MultinomialNB(),

    'gnb': GaussianNB(),

    'svm1': SVC(kernel='linear'),

    'svm2': SVC(kernel='rbf'),

    'svm3': SVC(kernel='sigmoid'),

    'mlp1': MLPClassifier(),

    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),

    'ada': AdaBoostClassifier(),

    'dtc': DecisionTreeClassifier(),

    'rfc': RandomForestClassifier(),

    'gbc': GradientBoostingClassifier(),

    'lr': LogisticRegression()

}
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

accuracies = dict()

for clf_name in clfs:

    clf = clfs[clf_name]

    clf.fit(X_train, y_train)

    accuracies[clf_name] = clf.score(X_valid, y_valid)
accuracies
# the kernel is RBF

parameters = {

    'C': [1, 10, 100],

    'gamma': [0.001, 0.01, 0.1]

}
svc = SVC()

clf = GridSearchCV(svc, parameters, scoring='accuracy', return_train_score=True)

clf.fit(X, y)
pd.DataFrame(clf.cv_results_)
parameters = {

    'hidden_layer_sizes': [

        [50,],

        [100,],

        [200,],

        [50, 50],

        [50, 100],

        [100, 50],

        [100, 100],

        [200, 200],

        [100, 100, 100]

    ]

}
mlp = MLPClassifier()

clf = GridSearchCV(mlp, parameters, scoring='accuracy', return_train_score=True)

clf.fit(X, y)
pd.DataFrame(clf.cv_results_)
parameters = {

    'activation': ['logistic', 'tanh', 'relu'],

    'solver': ['lbfgs', 'sgd', 'adam']

}
mlp = MLPClassifier(hidden_layer_sizes=[100, 100, 100])

clf = GridSearchCV(mlp, parameters, scoring='accuracy', return_train_score=True)

clf.fit(X, y)
pd.DataFrame(clf.cv_results_)
X = data.drop('Survived', axis=1).values

y = data['Survived'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

accuracies = dict()

for clf_name in clfs:

    clf = clfs[clf_name]

    clf.fit(X_train, y_train)

    accuracies[clf_name] = clf.score(X_valid, y_valid)
accuracies
parameters = {

    'n_estimators': [10, 50, 100, 150, 200],

    'criterion': ['gini', 'entropy'],

    'max_depth': [None, 5, 10],

    'bootstrap': [True, False]

}

rfc = RandomForestClassifier()

clf = GridSearchCV(rfc, parameters, scoring='accuracy', return_train_score=True)

clf.fit(X, y)

pd.DataFrame(clf.cv_results_)
X = feature_extraction(data)

y = data['Survived'].values
parameters = {

    'n_estimators': [10, 50, 100, 150, 200],

    'criterion': ['gini', 'entropy'],

    'max_depth': [None, 5, 10],

    'bootstrap': [True, False]

}

rfc = RandomForestClassifier()

clf = GridSearchCV(rfc, parameters, scoring='accuracy', return_train_score=True)

clf.fit(X, y)

pd.DataFrame(clf.cv_results_)
# re-read the data

data = pd.read_csv('../input/train.csv')

data.head()
data.count()
data['Name'].head(10)
titles = []

for name in data['Name'].str.lower():

    right_part = name.split(', ')[1]  # right_part = {title}. {first names}

    title = right_part.split('.')[0]

    titles.append(title)

print(len(titles))
titles = list(set(titles))

print(len(titles))

titles
titles = ['unk', 'miss', 'mlle', 'mrs', 'mr', 'ms', 'mme', 'lady', 'dr', 'sir', 'master', 'rev', 'major', 'col', 'capt', 'don', 'jonkheer', 'the countess']

len(titles)
title_encoder = LabelEncoder()

tmp = title_encoder.fit_transform(titles)

tmp
def get_title_from_name(name):

    right_part = name.split(', ')[1]  # right_part = {title}. {first names}

    title = right_part.split('.')[0]

    known_titles = ['unk', 'miss', 'mlle', 'mrs', 'mr', 'ms', 'mme', 'lady', 'dr', 'sir', 'master', 'rev', 'major', 'col', 'capt', 'don', 'jonkheer', 'the countess']

    return title if title in known_titles else 'unk'
def set_titles(df):

    titles = df['Name'].str.lower().apply(get_title_from_name)

    titles = np.array(titles)

    df['Title'] = titles
set_titles(data)

data.head()
def cabin_to_deck(cabin):

    if cabin == 'UNK':

        return 'UNK'

    return cabin[0]
data = pd.read_csv('../input/train.csv')
encoders, column_values = dict(), dict()
def preprocess(df, test=False):

    # fill in the missing values

    df['Age'].fillna(df['Age'].median(), inplace=True)

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    df['Cabin'].fillna('UNK', inplace=True)

    df['Embarked'].fillna('UNK', inplace=True)

    

    # create columns for feature engineering

    # add the 'Title' column

    set_titles(df)

    

    # add a 'FamilySize' column

    df['FamilySize'] = df['SibSp'] + df['Parch']

    

    # turn the cabin number into Deck

    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'UNK']

    df['Deck'] = df['Cabin'].apply(cabin_to_deck)

    

    # define the lists of column values for categorical data,

    # and the corresponding encoders (if train data)

    if not test:

        for column in ['Sex', 'Embarked', 'Deck']:

            column_values[column] = list(set(df[column])) + ['other']

            encoders[column] = LabelEncoder()

            encoders[column].fit(column_values[column])

    

    # adjust certain values if test data

    if test:

        for column in ['Sex', 'Embarked', 'Deck']:

            df[column] = df[column].apply(lambda x: x if x in column_values[column] else 'other')

    

    # drop the 'PassengerId', 'Name' and 'Cabin' columns

    df.drop('PassengerId', axis=1, inplace=True)

    df.drop('Name', axis=1, inplace=True)

    df.drop('Ticket', axis=1, inplace=True)

    df.drop('Cabin', axis=1, inplace=True)
preprocess(data)
def feature_extraction(df):

    # unchanged columns

    X = np.concatenate((df['Age'].values.reshape(-1, 1),

                        df['Pclass'].values.reshape(-1, 1),

                        df['SibSp'].values.reshape(-1, 1),

                        df['Parch'].values.reshape(-1, 1),

                        df['Fare'].values.reshape(-1, 1),

                        df['FamilySize'].values.reshape(-1, 1)),

                        axis=1)

    

    # ordinal encoding of Sex, Embarked and Deck

    for column in ['Sex', 'Embarked', 'Deck']:

        values = encoders[column].transform(df[column].values.reshape(-1, 1))

        values = np.array(values).reshape(-1, 1)

        X = np.concatenate((X, values), axis=1)

    

    # the other columns will not be used

    return X
X = feature_extraction(data)

y = data['Survived'].values
clfs = {

    'mnb': MultinomialNB(),

    'gnb': GaussianNB(),

    'svm1': SVC(kernel='linear'),

    'svm2': SVC(kernel='rbf'),

    'svm3': SVC(kernel='sigmoid'),

    'mlp1': MLPClassifier(),

    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),

    'ada': AdaBoostClassifier(),

    'dtc': DecisionTreeClassifier(),

    'rfc': RandomForestClassifier(),

    'gbc': GradientBoostingClassifier(),

    'lr': LogisticRegression()

}
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

accuracies = dict()

for clf_name in clfs:

    clf = clfs[clf_name]

    clf.fit(X_train, y_train)

    accuracies[clf_name] = clf.score(X_valid, y_valid)
accuracies
parameters = {

    'loss': ['deviance', 'exponential'],

    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],

    'n_estimators': [50, 100, 150, 200, 300],

    'max_depth': [3, 5, 8, 10]

}

gbc = GradientBoostingClassifier()

clf = GridSearchCV(gbc, parameters, scoring='accuracy', return_train_score=True)

clf.fit(X, y)

pd.DataFrame(clf.cv_results_)
print('Best score: {0}'.format(clf.best_score_))

print('Best params: {0}'.format(clf.best_params_))
# read and preprocess the whole train dataset

data = pd.read_csv('../input/train.csv')

encoders, column_values = dict(), dict()

preprocess(data)



# extract the features from the whole training dataset provided

X = feature_extraction(data)



# format the labels in a sutable way for a classifier

y = data['Survived'].values



# train the classifier

clf = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', max_depth=3, n_estimators=100)

clf.fit(X, y)
# read the test data

test_data = pd.read_csv('../input/test.csv')



# we'll save the passenger ids because we need them for the submission file

passenger_ids = test_data['PassengerId'].values



# preprocess the test data

preprocess(test_data, test=True)



# extract the features

X_test = feature_extraction(test_data)



# make the predictions

y_pred = clf.predict(X_test)
# save the submission in a file

submission = pd.DataFrame({

    'PassengerId': passenger_ids,

    'Survived': y_pred

})

submission.to_csv('submission.csv', index=False)

submission.head()