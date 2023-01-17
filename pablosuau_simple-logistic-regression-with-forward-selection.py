import sys

import random

from math import sqrt

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
df_training = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
len(df_training)
len(df_test)
df_training.head()
df_training.columns
df_training['Cabin'].unique()
df_training['Ticket'].values
def process_ticket(df):

    df['TicketPrefix'] = df['Ticket']

    df.loc[df['Ticket'].notnull(), 'TicketPrefix'] = df['Ticket'].apply(lambda x: x.split(' ')[0] 

                                                                                  if len(x.split(' ')) > 1

                                                                                  else 'NUMBER')

    

process_ticket(df_training)

process_ticket(df_test)
df_training[['Ticket', 'TicketPrefix']].head()
# For cabin I keep the first letter. There are multiple instances of rows having multiple assigned cabins. In these cases

# the first letter is the same for all the assigned cabins, except in two cases in which we have:

# F GXX

# In this case, for simplicity, I decided to keep the F letter

def process_cabin(df):

    df['CabinClass'] = df['Cabin']

    df.loc[df['Cabin'].notnull(), 'CabinClass'] = df['Cabin'].apply(lambda x: str(x)[0])

    

process_cabin(df_training)

process_cabin(df_test)
df_training[['Cabin', 'CabinClass']].head()
dependent = 'Survived'

categorical = ['Pclass', 'Sex', 'TicketPrefix', 'CabinClass', 'Embarked']

numerical = ['Age', 'SibSp', 'Parch', 'Fare']
kwargs = dict(histtype = 'stepfilled', alpha = 0.3, density = True, ec = 'k')



for n in numerical:

    df = df_training[df_training[n].notnull()]

    x = df[n].values

    y = df[dependent].values

    

    fig, ax = plt.subplots(1, 2)

    (_, bins, _) = ax[0].hist(x, **kwargs)

    ax[0].set_title(n)

    

    x_0 = x[np.where(y == 0)]

    x_1 = x[np.where(y == 1)]

    ax[1].hist(x_0, **kwargs, bins = bins)

    ax[1].hist(x_1, **kwargs, bins = bins)

    ax[1].legend(['no', 'yes'])

    ax[1].set_title(n + ' vs. survived')

    

    fig.set_figwidth(16)
for c in categorical:

    df = df_training[df_training[c].notnull()]

    

    fig, ax = plt.subplots(1, 2)

    freqs = df[c].value_counts()

    labels = freqs.keys()

    ax[0].bar(range(len(labels)), freqs.values, alpha = 0.3)

    ax[0].set_xticks(range(len(labels)))

    ax[0].set_xticklabels(labels, rotation = 'vertical')

    ax[0].set_title(c)

    

    freqs_01 = df.groupby('Survived')[c].value_counts()

    ax[1].bar(range(len(labels)), freqs_01[0][labels].values, alpha = 0.3)

    ax[1].bar(range(len(labels)), freqs_01[1][labels].values, bottom = freqs_01[0][labels].values, alpha = 0.3)

    ax[1].set_xticks(range(len(labels)))

    ax[1].set_xticklabels(labels, rotation = 'vertical')

    ax[1].legend(['no', 'yes'])

    ax[1].set_title(c + ' vs. survived')

    

    fig.set_figwidth(16)
def test_missing():

    for col in numerical + categorical:

        if col in categorical:

            missing = df_training[df_training[col].isna()]

        else:

            missing = df_training[(df_training[col].isna()) | 

                                  (df_training[col].apply(lambda x: type(x) == str))]

        proportion = len(missing) / len(df_training) * 100

        print(col + ': ' + str(proportion) + '%')
test_missing()
# Categorical variables

for c in ['CabinClass', 'Embarked']:

    df_training.loc[df_training[c].isna(), c] = 'None'

    df_test.loc[df_training[c].isna(), c] = 'None'
# Numerical variable

imputed = df_training[np.isreal(df_training['Age'])]['Age'].median()

df_training.loc[(df_training['Age'].isna()) | (~np.isreal(df_training['Age'])), 'Age'] = imputed

df_test.loc[(df_test['Age'].isna()) | (~np.isreal(df_test['Age'])), 'Age'] = imputed



# It turns out that the test data has a missing fare

imputed = df_training[np.isreal(df_training['Fare'])]['Fare'].median()

df_test.loc[(df_test['Fare'].isna()) | (~np.isreal(df_test['Fare'])), 'Fare'] = imputed
test_missing()
features = categorical + numerical



fig, ax = plt.subplots(6, 6)



plots = 0

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        row = int(plots / 6)

        col = plots % 6



        def categorical_to_numerical(f):

            if features[f] in numerical:

                values_f = df_training[features[f]]

            else:

                values = df_training[features[f]].unique()

                values_f = df_training[features[f]].values.copy()

                for v in range(len(values)):

                    values_f[np.where(values_f == values[v])] = v

            

            return values_f

        

        values_i = categorical_to_numerical(i)

        values_j = categorical_to_numerical(j)

        

        cor = ((values_i - values_i.mean()) * (values_j - values_j.mean()) / \

              ((len(values_i) - 1) * values_i.std() * values_j.std())).sum()

            

        ax[row][col].scatter(values_i, values_j, alpha = 0.5)

        

        ax[row][col].set_xlabel(features[i])

        ax[row][col].set_ylabel(features[j])

        ax[row][col].set_title('cor = ' + '%.2f' % cor)

        

        if features[i] in categorical:

            values = df_training[features[i]].unique().tolist()

            ax[row][col].set_xticks(range(len(values)))

            ax[row][col].set_xticklabels(values, rotation = 'vertical')

        if features[j] in categorical:

            values = df_training[features[j]].unique().tolist()

            ax[row][col].set_yticks(range(len(values)))

            ax[row][col].set_yticklabels(values)



        plots = plots + 1

        

fig.set_figwidth(16)

fig.set_figheight(16)

plt.tight_layout()
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
print(len(categorical + numerical))
variables = new_categorical + numerical

print(len(variables))
# Keeping these values to transform the test dataset

statistics = pd.concat((df_training.mean(), df_training.std()), axis = 1)

statistics.columns = ['mean', 'std']

statistics.head()
for c in variables:

    mean = statistics.loc[c, 'mean']

    std = statistics.loc[c, 'std']

    df_training[c] = (df_training[c] - mean) /  std

    df_test[c] = (df_test[c] - mean) /  std
df_training[variables].head()
# Removing columns

c = ['Name', 'Ticket', 'Cabin']

df_training = df_training.drop(c, axis = 1)

df_test = df_test.drop(c, axis = 1)
print(str((df_training.Survived == 1).sum()) + ' rows have Survived = 1')

print(str((df_training.Survived == 0).sum()) + ' rows have Survived = 0')
random.seed(0)
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
def evaluate(datasets, C = None):

    if C is None:

        C = 1

    logreg = LogisticRegression(solver = 'lbfgs', C = C)

    logreg.fit(datasets['X_train'], datasets['y_train'])

    y_pred = logreg.predict(datasets['X_test'])

    return sqrt(np.sum(np.power(np.array(y_pred) - np.array(datasets['y_test']), 2)))
def k_fold_cross_validation(df_training, folds, column_indexes, C = None):

    error = 0

    

    for k in range(10):

        train_indexes = []

        for j in range(10):

            if j == k:

                test_indexes = folds[j]

            else:

                train_indexes = train_indexes + folds[j]

                

        datasets = produce_training_test_set(df_training, train_indexes, test_indexes, column_indexes)

        

        error = error + evaluate(datasets, C)

        

    return error / 10.0
# RMSE if we use all the features

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
model_forward = model

columns = df_training.columns[model_forward]

X_train = df_training[columns].values

X_test = df_test[columns].values

y_train = df_training['Survived'].values
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X_train, y_train)
y_test = logreg.predict(X_test)
submission = df_test.copy()

submission['Survived'] = y_test

submission = submission[['PassengerId', 'Survived']]
submission.head()
submission.to_csv('logistic_regression_forward_selection.csv', index = False)