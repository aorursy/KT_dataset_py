import sys

import math

import random

from math import sqrt

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
random.seed(0)
# Reading the input data

df_training = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# Preprocess the ticket field

def process_ticket(df):

    df['TicketPrefix'] = df['Ticket']

    df.loc[df['Ticket'].notnull(), 'TicketPrefix'] = df['Ticket'].apply(lambda x: x.split(' ')[0] 

                                                                                  if len(x.split(' ')) > 1

                                                                                  else 'NUMBER')

    

process_ticket(df_training)

process_ticket(df_test)
# For cabin I keep the first letter. There are multiple instances of rows having multiple assigned cabins. In these cases

# the first letter is the same for all the assigned cabins, except in two cases in which we have:

# F GXX

# In this case, for simplicity, I decided to keep the F letter

def process_cabin(df):

    df['CabinClass'] = df['Cabin']

    df.loc[df['Cabin'].notnull(), 'CabinClass'] = df['Cabin'].apply(lambda x: str(x)[0])

    

process_cabin(df_training)

process_cabin(df_test)
# Imputing missing categorical variables

for c in ['CabinClass', 'Embarked']:

    df_training.loc[df_training[c].isna(), c] = 'None'

    df_test.loc[df_training[c].isna(), c] = 'None'
# Imputing missing numerical variables

imputed = df_training[np.isreal(df_training['Age'])]['Age'].median()

df_training.loc[(df_training['Age'].isna()) | (~np.isreal(df_training['Age'])), 'Age'] = imputed

df_test.loc[(df_test['Age'].isna()) | (~np.isreal(df_test['Age'])), 'Age'] = imputed



# It turns out that the test data has a missing fare

imputed = df_training[np.isreal(df_training['Fare'])]['Fare'].median()

df_test.loc[(df_test['Fare'].isna()) | (~np.isreal(df_test['Fare'])), 'Fare'] = imputed
dependent = 'Survived'

categorical = ['Pclass', 'Sex', 'TicketPrefix', 'CabinClass', 'Embarked']

numerical = ['Age', 'SibSp', 'Parch', 'Fare']
# Processing categorical variables

new_categorical = []

for c in categorical:

    values = df_training[c].unique()[:-1]

    for v in values:

        name = c + '_' + str(v)

        df_training[name] = (df_training[c] == v).astype(int)

        df_test[name] = (df_test[c] == v).astype(int)

        new_categorical.append(name)

    df_training = df_training.drop(c, axis = 1)

    df_test = df_test.drop(c, axis = 1)



variables = new_categorical + numerical
# Standardising variables

statistics = pd.concat((df_training.mean(), df_training.std()), axis = 1)

statistics.columns = ['mean', 'std']



for c in variables:

    mean = statistics.loc[c, 'mean']

    std = statistics.loc[c, 'std']

    df_training[c] = (df_training[c] - mean) /  std

    df_test[c] = (df_test[c] - mean) /  std
# Removing redundant columns

c = ['Name', 'Ticket', 'Cabin']

df_training = df_training.drop(c, axis = 1)

df_test = df_test.drop(c, axis = 1)
df_training.head()
df_test.head()
# Dimnesionality reduction of the training data

X = df_training.values[:, 2:]

pca = PCA(n_components = 2)

X_pca = pca.fit_transform(X)
index_1 = df_training.Survived == 1

index_0 = df_training.Survived == 0
fig, ax = plt.subplots()

ax.scatter(X_pca[index_1, 0], X_pca[index_1, 1], c = 'blue', alpha = 0.5)

ax.scatter(X_pca[index_0, 0], X_pca[index_0, 1], c = 'red', alpha = 0.5)

_ = ax.legend(['Survived', 'Not survived'])
X_test_pca = pca.transform(df_test.values[:, 1:])
fig, ax = plt.subplots()

ax.scatter(X_pca[index_1, 0],  X_pca[index_1, 1], c = 'blue', alpha = 0.5)

ax.scatter(X_pca[index_0, 0], X_pca[index_0, 1], c = 'red', alpha = 0.5)

ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c = 'gray', alpha = 0.5)

_ = ax.legend(['Survived', 'Not survived', 'Test'])
# Estimating the parameters of the 'Survived' and 'Not survived' classes

mean_0 = np.mean(X_pca[index_0, :], axis = 0)

mean_1 = np.mean(X_pca[index_1, :], axis = 0)

cov_0 = np.cov(X_pca[index_0, :].T)

cov_1 = np.cov(X_pca[index_1, :].T)
def mahalanobis(p, mean, cov):

    dif = p - mean

    return math.sqrt(np.dot(np.dot(dif.T, np.linalg.inv(cov)), dif))
# Predictions for the test data

y_test = []

for i in range(X_test_pca.shape[0]):

    dist_0 = mahalanobis(X_test_pca[i, :], mean_0, cov_0)

    dist_1 = mahalanobis(X_test_pca[i, :], mean_1, cov_1)

    y_test.append(int(dist_1 < dist_0))
submission = df_test.copy()

submission['Survived'] = y_test

submission = submission[['PassengerId', 'Survived']]
submission.head()
# generating sets for 10-fold cross validation

indexes = list(range(len(df_training)))

random.shuffle(indexes)

folds = []

for i in range(10):

    folds.append([])

for i in range(len(indexes)):

    folds[i % 10].append(indexes[i])
def produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes):

    columns = df_training.columns[column_indexes]

    datasets = {}

    datasets['X_train'] = df_training.iloc[train_indexes][columns].values

    datasets['X_test'] = df_training.iloc[test_indexes][columns].values

    datasets['y_train'] = df_training.iloc[train_indexes]['Survived'].values

    datasets['y_test'] = df_training.iloc[test_indexes]['Survived'].values

    

    return datasets
def evaluate(datasets):

    if datasets['X_train'].shape[1] > 2:

        pca = PCA(n_components = 2)

        X_training_pca = pca.fit_transform(datasets['X_train'])

        X_test_pca = pca.transform(datasets['X_test'])

    else:

        X_training_pca = datasets['X_train']

        X_test_pca = datasets['X_test']

        

    mean_0 = np.mean(X_training_pca[datasets['y_train'] == 0, :], axis = 0)

    mean_1 = np.mean(X_training_pca[datasets['y_train'] == 1, :], axis = 0)

    cov_0 = np.cov(X_training_pca[datasets['y_train'] == 0, :].T)

    cov_1 = np.cov(X_training_pca[datasets['y_train'] == 1, :].T)

    

    y_pred = []

    for i in range(X_test_pca.shape[0]):

        if datasets['X_train'].shape[1] > 1:

            dist_0 = mahalanobis(X_test_pca[i, :], mean_0, cov_0)

            dist_1 = mahalanobis(X_test_pca[i, :], mean_1, cov_1)

        else:

            dist_0 = (X_test_pca[i, :] - mean_0) / cov_0

            dist_1 = (X_test_pca[i, :] - mean_1) / cov_1

        y_pred.append(int(dist_1 < dist_0))

        

    return sqrt(np.sum(np.power(np.array(y_pred) - np.array(datasets['y_test']), 2)))
def k_fold_cross_validation(df_training, folds, column_indexes):

    error = 0

    

    for k in range(10):

        train_indexes = []

        for j in range(10):

            if j == k:

                test_indexes = folds[j]

            else:

                train_indexes = train_indexes + folds[j]

                

        datasets = produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes)

        

        error = error + evaluate(datasets)

        

    return error / 10.0
column_indexes = list(range(2, 62))

k_fold_cross_validation(df_training, folds, column_indexes)
# Forward selection

pending = list(range(2, 62))

model = []

min_error = sys.float_info.max

while len(pending) > 0:

    

    prev_error = min_error

    min_error = sys.float_info.max

    

    for i in pending:

        new_model = model + [i]

        error = k_fold_cross_validation(df_training, folds, new_model)

        

        if error < min_error:

            min_error = error

            best_model = new_model

            feature = i

            

    if min_error < prev_error:

        print('Selecting feature ' + df_training.columns[feature] + ' - error decreased to ' + str(min_error))

        model = best_model

        pending.remove(feature)

    else:

        print('END')

        break
columns = df_training.columns[model]



pca = PCA(n_components = 2)

X_training_pca = pca.fit_transform(df_training[columns].values)

X_test_pca = pca.transform(df_test[columns].values)



mean_0 = np.mean(X_training_pca[index_0, :], axis = 0)

mean_1 = np.mean(X_training_pca[index_1, :], axis = 0)

cov_0 = np.cov(X_training_pca[index_0, :].T)

cov_1 = np.cov(X_training_pca[index_1, :].T)



y_pred = []

for i in range(X_test_pca.shape[0]):

    dist_0 = mahalanobis(X_test_pca[i, :], mean_0, cov_0)

    dist_1 = mahalanobis(X_test_pca[i, :], mean_1, cov_1)

    y_pred.append(int(dist_1 < dist_0))
fig, ax = plt.subplots()

ax.scatter(X_training_pca[index_1, 0], X_training_pca[index_1, 1], c = 'blue', alpha = 0.5)

ax.scatter(X_training_pca[index_0, 0], X_training_pca[index_0, 1], c = 'red', alpha = 0.5)

ax.set_yscale('symlog')

_ = ax.legend(['Survived', 'Not survived', 'Test'])
submission = df_test.copy()

submission['Survived'] = y_pred

submission = submission[['PassengerId', 'Survived']]
submission.head()
submission.to_csv('mahalanobis_forward_selection.csv', index = False)