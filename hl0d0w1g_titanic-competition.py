# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import itertools as it

import lightgbm as lgb

import re

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_curve

import random



RANDOM_SEED = 0



sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load and explore labelled data

df_data = pd.read_csv('/kaggle/input/titanic/train.csv')



print(df_data.info())

df_data.describe()
df_data
target_col = 'Survived'



# Split data to get validation set

df_train, df_val = train_test_split(df_data, test_size=0.2, random_state=RANDOM_SEED)



train_X, train_y = df_train.drop(target_col, axis='columns'), df_train[target_col]

val_X, val_y = df_val.drop(target_col, axis='columns'), df_val[target_col]
survived_vc = train_y.value_counts()

print(survived_vc)

print('1s: {0:.2f}%'.format(train_y.mean() * 100))

ax = sns.barplot(survived_vc.index, survived_vc.values)

plt.title('Train')

plt.show()
def drop_cols(X):

    cols = ['PassengerId', 'Ticket']

    X = X.drop(cols, axis='columns')

    

    return X



train_X = drop_cols(train_X)

val_X = drop_cols(val_X)
class NameProcessor():

    def __init__(self):

        self._col = 'Name'

        self._encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        

        return



    def preprocess_name(self, X):

        if not len(set(X[self._col])) == 5:

            X[self._col] = X[self._col].str.extract(' ([A-Za-z]+)\.', expand=False)

            X[self._col] = X[self._col].replace('Mlle', 'Miss')

            X[self._col] = X[self._col].replace('Ms', 'Miss')

            X[self._col] = X[self._col].replace('Mme', 'Mrs')

            X[self._col] = X[self._col].replace(['Lady', 'Countess', 'Capt', 'Don', 'Col', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Unknown')

        

        return X

    

    def fit(self, X, y=None):

        X = self.preprocess_name(X)

        self._encoder.fit(X[self._col].values.reshape(-1, 1))

        

        return self 

    

    def transform(self, X, y=None):

        X = self.preprocess_name(X)

        OH_cols_X = pd.DataFrame(self._encoder.transform(X[self._col].values.reshape(-1, 1)))

        OH_cols_X.index = X.index

        X = X.drop([self._col], axis=1)

        X = pd.concat([X, OH_cols_X], axis='columns')

        

        return X



name_processor = NameProcessor()

name_processor.fit(train_X)



train_X = name_processor.transform(train_X)

val_X = name_processor.transform(val_X)
def process_sex(X):

    col = 'Sex'

    sex_cat = {'male': 1, 'female': 0}



    X[col] = X[col].map(sex_cat)

    

    return X



train_X = process_sex(train_X)

val_X = process_sex(val_X)
class AgeProcessor():

    def __init__(self):

        self._col = 'Age'

        self._imputer = SimpleImputer(strategy='mean')

        

        return



    def fit(self, X, y=None):

        self._imputer.fit(X[self._col].values.reshape(-1, 1))

        

        return self 

    

    def transform(self, X, y=None):

        imputed_X = pd.DataFrame(self._imputer.transform(X[self._col].values.reshape(-1, 1)))

        imputed_X.columns = [self._col]

        X[self._col] = imputed_X.values

        X[self._col] = X[self._col].astype(int)

        

        return X



age_processor = AgeProcessor()

age_processor.fit(train_X)



train_X = age_processor.transform(train_X)

val_X = age_processor.transform(val_X)
class FareProcessor():

    def __init__(self):

        self._col = 'Fare'

        self._class_col = 'Pclass'

        self._class_mean_fare = {}

        

        return



    def fit(self, X, y=None):

        for p_class in set(X[self._class_col]):

            self._class_mean_fare[p_class] = X.loc[X[self._class_col] == p_class][self._col].dropna().mean()



        return self 

    

    def transform(self, X, y=None):

        X[self._col] = X.apply(lambda row: self._class_mean_fare[row[self._class_col]] if np.isnan(row[self._col]) else row[self._col], axis='columns')

        

        return X



fare_processor = FareProcessor()

fare_processor.fit(train_X)



train_X = fare_processor.transform(train_X)

val_X = fare_processor.transform(val_X)
def process_cabin(X):

    col = 'Cabin'

    deck_cat = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 0}



    X[col] = X[col].fillna("U0")

    X[col] = X[col].apply(lambda s: re.compile("([a-zA-Z]+)").search(s).group())

    X[col] = X[col].map(deck_cat)

    X[col] = X[col].fillna(0)

    X[col] = X[col].astype(int)



    #dataset['Room_Number'] = dataset['Cabin'].apply(lambda s: re.compile("([0-9]+)").search(s).group())

    #dataset['Room_Number'] = dataset['Room_Number'].astype(int)



    return X



train_X = process_cabin(train_X)

val_X = process_cabin(val_X)
class EmbarkedProcessor():

    def __init__(self):

        self._col = 'Embarked'

        self._imputer = SimpleImputer(strategy='most_frequent')

        self._encoder = LabelEncoder()

        

        return



    def fit_imputer(self, X, y=None):

        self._imputer.fit(X[self._col].values.reshape(-1, 1))

        

        return self 

    

    def transform_imputer(self, X, y=None):

        imputed_X = pd.DataFrame(self._imputer.transform(X[self._col].values.reshape(-1, 1)))

        imputed_X.columns = [self._col]

        X[self._col] = imputed_X.values

        

        return X

    

    def fit_encoder(self, X, y=None):

        self._encoder.fit(X[self._col])

        

        return self 

    

    def transform_encoder(self, X, y=None):

        X[self._col] = self._encoder.transform(X[self._col])

        

        return X



embarked_processor = EmbarkedProcessor()



embarked_processor.fit_imputer(train_X)

train_X = embarked_processor.transform_imputer(train_X)

val_X = embarked_processor.transform_imputer(val_X)



embarked_processor.fit_encoder(train_X)

train_X = embarked_processor.transform_encoder(train_X)

val_X = embarked_processor.transform_encoder(val_X)
train_dataset = lgb.Dataset(train_X, label=train_y)

val_dataset = lgb.Dataset(val_X, label=val_y)
param_grid = { 'task': ['train']

             , 'boosting': ['gbdt']

             , 'num_leaves': [20, 50, 75, 100]

             , 'objective': ['binary']

             , 'learning_rate': [0.1, 0.5] 

             , 'num_iterations': [1000]

             , 'max_depth': [10, 20, 40, 50]

             , 'min_data_in_leaf': [4, 5, 6] 

             , 'drop_rate': [0.1, 0.2, 0.3] 

             , 'early_stopping_rounds': [10]

             , 'seed': [1, 2, 10, 100] 

             }



hyperparams_keys = sorted(param_grid)

combinations = list(it.product(*(param_grid[dict_key] for dict_key in hyperparams_keys)))

combinations = list(map(lambda hp_config: dict(zip(hyperparams_keys, hp_config)), combinations))

# Se cambia el orden en el que se han producido las combinaciones de hiperparámetros

random.shuffle(combinations)

print(len(combinations))
def get_auroc(objetive, preds):

    '''

    Calculate area under ROC curve

    

    '''

    fpr, tpr, thresholds = roc_curve(objetive, preds)

    auroc = auc(fpr, tpr)

    

    return auroc



def get_auprc(objetive, preds):

    '''

    Calculate area under precision-recall curve

    

    '''

    precision, recall, thresholds = precision_recall_curve(objetive, preds)

    auprc = auc(recall, precision)

    

    return auprc
results = []

for hps in combinations:

    params = hps.copy()

    params.pop('num_iterations')

    params.pop('early_stopping_rounds')

    params['verbosity'] = -1

    params['num_threads'] = 1

    

    model = lgb.train(params, train_dataset, hps['num_iterations'], verbose_eval=0, valid_sets=[val_dataset], early_stopping_rounds=hps['early_stopping_rounds'])



    train_preds = model.predict(train_X, num_iteration=model.best_iteration)

    val_preds = model.predict(val_X, num_iteration=model.best_iteration)

    

    train_auroc = get_auroc(train_y, train_preds)

    val_auroc = get_auroc(val_y, val_preds)

    train_auprc = get_auprc(train_y, train_preds)

    val_auprc = get_auprc(val_y, val_preds)

    

    results.append({'hps': hps, 'train_auroc': train_auroc, 'val_auroc': val_auroc, 'train_auprc': train_auprc, 'val_auprc': val_auprc})

    

df_results = pd.DataFrame(results)

df_results = df_results.sort_values('val_auprc', ascending=False)

df_results
# Best model

hps = df_results.iloc[0]['hps']



X = pd.concat([train_X, val_X], axis=0, ignore_index=True)

y = pd.concat([train_y, val_y], axis=0, ignore_index=True)

dataset = lgb.Dataset(X, label=y)



params = hps.copy()

params.pop('num_iterations')

params.pop('early_stopping_rounds')

params['verbosity'] = -1

params['num_threads'] = 1



model = lgb.train(params, dataset, hps['num_iterations'], verbose_eval=0)



preds = model.predict(X, num_iteration=model.best_iteration)



auroc = get_auroc(y, preds)

auprc = get_auprc(y, preds)

print('AUROC: {}'.format(auroc))

print('AUPRC: {}'.format(auprc))
fpr, tpr, thresholds = roc_curve(y, preds)

auroc = auc(fpr, tpr)



auprc_random = 0.5



plt.figure(figsize=(8,8))

sns.set()

lw = 2

sns.lineplot(fpr, tpr, color='darkorange',

lw=lw, label='AUROC curve (area = %0.2f)' % auroc)

plt.plot([0.0, 1.0], [0.0, 1.0], color='navy', linestyle='--',

lw=lw, label='random AUROC curve (area = %0.2f)' % auprc_random)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('1 - Specificity', fontsize=18)

plt.ylabel('Sensitivity', fontsize=18)

plt.title('Roc curve', fontsize=18)

plt.legend(loc="lower right", fontsize=18)

plt.show()
precision, recall, thresholds = precision_recall_curve(y, preds)

auprc = auc(recall, precision)



precision_random, recall_random, thresholds_random = precision_recall_curve(y, np.random.rand(len(preds)))

auprc_random = auc(recall_random, precision_random)



plt.figure(figsize=(8,8))

sns.set()

lw = 2

sns.lineplot(recall, precision, color='darkorange',

lw=lw, label='AUPRC curve (area = %0.2f)' % auprc)

plt.plot([0, 1], [auprc_random, auprc_random], color='navy', linestyle='--',

lw=lw, label='random AUPRC curve (area = %0.2f)' % auprc_random)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('Recall', fontsize=18)

plt.ylabel('Precision', fontsize=18)

plt.title('Precision-Recall curve', fontsize=18)

plt.legend(loc="lower right", fontsize=18)

plt.show()
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



print(df_test.info())

df_test.describe()
test_X = df_test.copy()

test_X = drop_cols(test_X)

test_X = name_processor.transform(test_X)

test_X = process_sex(test_X)

test_X = age_processor.transform(test_X)

test_X = fare_processor.transform(test_X)

test_X = process_cabin(test_X)

test_X = embarked_processor.transform_imputer(test_X)

test_X = embarked_processor.transform_encoder(test_X)

print(test_X.info())



test_preds = model.predict(test_X, num_iteration=model.best_iteration)



test_submit = df_test[['PassengerId']]

test_submit['Survived'] = np.round(test_preds).astype(int)

test_submit
test_submit.to_csv('submission.csv', index=False)