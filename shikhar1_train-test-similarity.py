import matplotlib

%matplotlib inline

import seaborn as sns

sns.set()

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold as SKF

from sklearn.linear_model import LogisticRegression as LR

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score as AUC

from sklearn.model_selection import train_test_split

import pylab as plt

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.ensemble import forest

from sklearn.tree import export_graphviz
#loading test and train data

train = pd.read_csv('../input/unzippedraw/train.csv',low_memory=True)

test = pd.read_csv('../input/unzippedraw/test.csv',low_memory=True)
#making copy

trn =  train.copy()

tst =  test.copy()
#adding a column to identify whether a row comes from train or not

tst['is_train'] = 0

trn['is_train'] = 1 #1 for train
tst.shape,trn.shape
#combining test and train data

df_combine = pd.concat([trn, tst], axis=0, ignore_index=True)

#dropping 'target' column as it is not present in the test

df_combine = df_combine.drop('target', axis=1)
y = df_combine['is_train'].values #labels

x = df_combine.drop('is_train', axis=1).values #covariates or our dependent variables
tst, trn = tst.values, trn.values
def set_rf_samples(n):

    """ Changes Scikit learn's random forests to give each tree a random sample of

    n random rows.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))
def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))
set_rf_samples(60000) 

# reset_rf_samples() to revert back to default behavior
m = RandomForestClassifier(n_jobs=-1,max_depth=5)

predictions = np.zeros(y.shape)
skf = SKF(n_splits=20, shuffle=True, random_state=100)

for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):

    

    X_train, X_test = x[train_idx], x[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

        

    m.fit(X_train, y_train)

    probs = m.predict_proba(X_test)[:, 1]

    predictions[test_idx] = probs
print('ROC-AUC for X and Z distributions:', AUC(y, predictions))
plt.figure(figsize=(20,10))

predictions_train = predictions[len(tst):] #filtering the actual training rows

weights = (1./predictions_train) - 1. 

weights /= np.mean(weights) # Normalizing the weights

plt.xlabel('Computed sample weight')

plt.ylabel('# Samples')

sns.distplot(weights, kde=False)