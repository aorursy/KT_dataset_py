import sys

import random

import itertools

import multiprocessing

from math import sqrt

import pandas as pd

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
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
def evaluate(datasets, neigs, weights):

    clf = KNeighborsClassifier(n_neighbors = neigs, weights = weights)

    clf.fit(datasets['X_train'], datasets['y_train'])

    y_pred = clf.predict(datasets['X_test'])

    return sqrt(np.sum(np.power(np.array(y_pred) - np.array(datasets['y_test']), 2)))
def k_fold_cross_validation(df_training, folds, column_indexes, neigs, weights):

    error = 0

    

    for k in range(10):

        train_indexes = []

        for j in range(10):

            if j == k:

                test_indexes = folds[j]

            else:

                train_indexes = train_indexes + folds[j]

                

        datasets = produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes)

        

        error = error + evaluate(datasets, neigs, weights)

        

    return error / 10.0
K = range(1, 200)

W = ['uniform', 'distance']

column_indexes = list(range(2, 62)) # All columns

minimum = sys.float_info.max



errors = dict()

for w in W:

    errors[w] = list()

    for k in tqdm(K):

        error = k_fold_cross_validation(df_training, folds, column_indexes, k, w)

        errors[w].append(error)

        if error < minimum:

            minimum = error

            min_k = k

            min_w = w

            

print('Minimum for w = ' + min_w + ' and k = '+ str(min_k))
fig, ax = plt.subplots()

for w in W:

    ax.plot(K, errors[w])

ax.set_xlabel('k')

ax.set_ylabel('error')

ax.legend(W)

fig.set_figwidth(16)
def k_fold_cross_validation_unpack(args):

    return k_fold_cross_validation(*args)
# Forward selection

pending = list(range(2, 62))

model = []

min_error = sys.float_info.max

parameters = list(itertools.product(K, W))

num_processes = multiprocessing.cpu_count()

pool = multiprocessing.Pool(processes = num_processes)



while len(pending) > 0:

    prev_error = min_error

    min_error = sys.float_info.max

    

    for i in pending:

        new_model = model + [i]

        parameters = itertools.product([df_training], [folds], [new_model], K, W)

        

        errors = pool.map(k_fold_cross_validation_unpack, parameters)

        

        best = list(itertools.product(K, W))[np.argmin(errors)]

        minimum = min(errors)

        

        if minimum < min_error:

            min_error = minimum

            best_model = new_model

            feature = i

            best_k = best[0]

            best_w = best[1]

            

    if min_error < prev_error:

        print('Selecting feature ' + 

              df_training.columns[feature] + 

              '(k = ' + 

              str(best_k) + 

              ', w = ' +

              best_w + 

              ') - error decreased to ' +

              str(min_error))

        model = best_model

        pending.remove(feature)

    else:

        print('END')

        break



pool.close()
model_forward = model

columns = df_training.columns[model_forward]

X_train = df_training[columns].values

X_test = df_test[columns].values

y_train = df_training['Survived'].values
clf = KNeighborsClassifier(n_neighbors = best_k, weights = best_w)
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
submission = df_test.copy()

submission['Survived'] = y_test

submission = submission[['PassengerId', 'Survived']]
submission.head()
submission.to_csv('knn_forward_selection.csv', index = False)