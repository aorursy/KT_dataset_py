import numpy as np

import pandas as pd

import scipy

import sklearn
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
df_train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv', index_col='id')

df_test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv', index_col='id')
sample_submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')
binvar = ['bin_' + str(i) for i in range(1,5)]

ordvar = ['ord_' + str(i) for i in range(6)]

nomvar = ['nom_' + str(i) for i in range(10)]

dmvar  = ['day', 'month']
df_work = df_train.copy()

y_train = df_work['target'].copy()

df_work = pd.concat([df_work.drop('target', axis = 1), df_test])
df_work.drop('bin_0', inplace=True, axis=1)
df_work['bin_3'] = df_work['bin_3'].map({'F':0, 'T':1})

df_work['bin_4'] = df_work['bin_4'].map({'N':0, 'Y':1})
df_work['ord_0'] = df_work['ord_0'] - 1
ord1dict = {'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4}

df_work['ord_1'] = df_work['ord_1'].map(ord1dict)
ord2dict = {'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot':4, 'Lava Hot':5}

df_work['ord_2'] = df_work['ord_2'].map(ord2dict)
oe = OrdinalEncoder(categories='auto')

df_work[ordvar[3:]] = oe.fit_transform(df_work[ordvar[3:]])

for var, cl in zip(ordvar[3:], oe.categories_):

    print(var)

    print(cl)
df_work[ordvar] = StandardScaler().fit_transform(df_work[ordvar])
df_work[nomvar[5:]].nunique()
df_work['nom_5'] = df_work['nom_5'].str[4:]

df_work['nom_6'] = df_work['nom_6'].str[3:]

df_work['nom_7'] = df_work['nom_7'].str[3:]

df_work['nom_8'] = df_work['nom_8'].str[3:]

df_work['nom_9'] = df_work['nom_9'].str[3:]
df_work[nomvar[5:]].nunique()
enc = OneHotEncoder(categories = 'auto', dtype = 'float64', drop = 'first')

nom_matrix = enc.fit_transform(df_work[nomvar])

df_work.drop(nomvar, inplace=True, axis=1)
enc = OneHotEncoder(categories='auto', dtype = 'float64', drop = 'first')

dm_matrix = enc.fit_transform(df_work[dmvar])

df_work.drop(dmvar, inplace=True, axis=1)
df_work.columns
df_work_sprs =scipy.sparse.hstack([nom_matrix,

                                   scipy.sparse.coo_matrix(df_work).astype('float64'),

                                   dm_matrix]).tocsr()

display(df_work_sprs)
X_train = df_work_sprs[:y_train.shape[0]]

X_test = df_work_sprs[y_train.shape[0]:]
%%time



C = 0.12



clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, verbose=0, n_jobs=-1)





score = cross_validate(clf, X_train, y_train, cv=3, scoring="roc_auc")

mean = score['test_score'].mean()

print(score['test_score'])

print('C =', C, f'{mean:.8f}')
%%time



clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, verbose=0, n_jobs=-1)

clf.fit(X_train, y_train)
y_preds = clf.predict_proba(X_test)[:,1]
sample_submission['target'] = y_preds

sample_submission.to_csv('submission.csv')