import sys

import random

import itertools

import multiprocessing

from math import sqrt

import pandas as pd

import numpy as np

from tqdm import tqdm

from  warnings import simplefilter

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.exceptions import ConvergenceWarning
random.seed(0)

simplefilter('ignore', category = ConvergenceWarning)
# Reading the input data

df_training = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
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
def evaluate(datasets, C, kernel, kernel_params):

    kwargs = {}

    if kernel == 'poly':

        kwargs['degree'] = kernel_params['degree']

        kwargs['coef0'] = kernel_params['coef0']

       

    clf = SVC(max_iter = 50000, C = C, gamma = 'auto', kernel = kernel, **kwargs)

    clf.fit(datasets['X_train'], datasets['y_train'])

    y_pred = clf.predict(datasets['X_test'])

    return sqrt(np.sum(np.power(np.array(y_pred) - np.array(datasets['y_test']), 2)))
def k_fold_cross_validation(df_training, folds, column_indexes, C, kernel, kernel_params):

    error = 0

    

    for k in range(10):

        train_indexes = []

        for j in range(10):

            if j == k:

                test_indexes = folds[j]

            else:

                train_indexes = train_indexes + folds[j]

                

        datasets = produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes)

        

        error = error + evaluate(datasets, C, kernel, kernel_params)

        

    return error / 10.0
# Results were very similar for gamma = scale and gamma = auto

# No good results with degree = 1

# Manually setting the ranges and steps after testing multiple times so we do not get so many combinations

# for the feature selection case

C = np.arange(0.1, 5.1, 0.1).tolist()

kernel = ['linear', 'poly', 'rbf']

degree = [2, 3]

coef0 = np.arange(0, 3.2, 0.2).tolist()



poly_params = list(itertools.product(*[degree, coef0]))



comb = list(itertools.product(*[C, ['linear'], [None], [None]]))

comb.extend(list(itertools.product(*[C, ['rbf'], [None], [None]])))

comb.extend(list(itertools.product(*[C, ['poly'], degree, coef0])))



column_indexes = list(range(2, 62)) # All columns

minimum = sys.float_info.max



errors = pd.DataFrame(data = comb, columns = ['C', 'kernel', 'degree', 'coef0'])

errors['error'] = np.nan



for i in tqdm(range(len(errors))):

    errors.loc[i, 'error'] = k_fold_cross_validation(df_training,

                                                     folds,

                                                     column_indexes,

                                                     errors['C'].loc[i],

                                                     errors['kernel'].loc[i],

                                                     {'degree': errors['degree'].loc[i],

                                                      'coef0': errors['coef0'].loc[i]})
errors = errors.sort_values(by = 'error')

errors.head(5)
fig, ax = plt.subplots()

errors_linear = errors[errors.kernel == 'linear'].sort_values(by = 'C')

ax.plot(errors_linear.C, errors_linear.error)

ax.set_xlabel('C')

ax.set_ylabel('RMSE')

ax.set_title('Linear model')

ax.grid(True)
fig, ax = plt.subplots()

errors_rbf = errors[errors.kernel == 'rbf'].sort_values(by = 'C')

ax.plot(errors_rbf.C, errors_rbf.error)

ax.set_xlabel('C')

ax.set_ylabel('RMSE')

ax.set_title('RBF kernel')

ax.grid(True)
fig, ax = plt.subplots(1, len(degree))

errors_poly = errors[errors.kernel == 'poly']

for d in degree:

    i = degree.index(d)

    errors_d = errors_poly[errors_poly.degree == d].pivot(index='C', 

                                                          columns='coef0', 

                                                          values='error')

    im = ax[i].imshow(errors_d, cmap = 'viridis', extent=[errors.C.min(), 

                                                          errors.C.max(), 

                                                          errors.coef0.min(), 

                                                          errors.coef0.max()])

    fig.colorbar(im, ax = ax[i])

    ax[i].set_xlabel('C')

    ax[i].set_ylabel('coef0')

    ax[i].set_title('poly kernel - degree = ' + str(d))

fig.set_figwidth(12)

fig.set_figheight(6)
best_C = errors.C.values[0]

best_kernel = errors.kernel.values[0]

best_degree = errors.degree.values[0]

best_coef0 = errors.coef0.values[0]

clf = SVC(C = best_C, 

          gamma = 'auto', 

          kernel = best_kernel, 

          degree = best_degree,

          coef0 = best_coef0)
columns = df_training.columns[2:]

X_train = df_training[columns].values

X_test = df_test[columns].values

y_train = df_training['Survived'].values
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
submission = df_test.copy()

submission['Survived'] = y_test

submission = submission[['PassengerId', 'Survived']]
submission.head()
submission.to_csv('svm.csv', index = False)