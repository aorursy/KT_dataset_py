import numpy as np

import pandas as pd

import os

import scipy

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
train.head()
test.head()
train.nunique()
train.isna().sum()
bin_cols = [col  for col in train.columns.values if col.startswith('bin')]

nom_cols = [col  for col in train.columns.values if col.startswith('nom')]

ord_cols = [col  for col in train.columns.values if col.startswith('ord')]
def ord_to_num(df, col):

    keys=np.sort(df[col].unique())

    values=np.arange(len(keys))

    map = dict(zip(keys, values))

    df[col] = df[col].replace(map)
ord_to_num(test, 'ord_3')

ord_to_num(train, 'ord_3')
ord_to_num(train, 'ord_4')

ord_to_num(test,'ord_4')
ord_to_num(train, 'ord_5')

ord_to_num(test,'ord_5')
keys_ord_1=train.ord_1.unique()

keys_ord_1
values_ord_1=[3,4,0,1,2]
map_ord_1 = dict(zip(keys_ord_1, values_ord_1))

map_ord_1
train['ord_1'] = train['ord_1'].replace(map_ord_1)

test['ord_1'] = test['ord_1'].replace(map_ord_1)
keys_ord_2=train.ord_2.unique()

keys_ord_2
values_ord_2=[1,3,5,4,0,2]
map_ord_2 = dict(zip(keys_ord_2, values_ord_2))

map_ord_2
train['ord_2'] = train['ord_2'].replace(map_ord_2)

test['ord_2'] = test['ord_2'].replace(map_ord_2)
train[ord_cols].head()
train[ord_cols].nunique()
train['ord_4_band'] = pd.qcut(train['ord_4'], 6)

bands=train.ord_4_band.unique()

keys_bands=np.sort(bands)

values_bands=np.arange(len(keys_bands))

map_bands = dict(zip(keys_bands, values_bands))
train['ord_4_band'] = train['ord_4_band'].replace(map_bands)

test['ord_4_band']=pd.cut(test.ord_4,pd.IntervalIndex(keys_bands))
test['ord_4_band'] = test['ord_4_band'].replace(map_bands)

test.ord_4_band.head()
train['ord_5_band'] = pd.qcut(train['ord_5'], 6)

bands=train.ord_5_band.unique()

keys_bands=np.sort(bands)

values_bands=np.arange(len(keys_bands))

map_bands = dict(zip(keys_bands, values_bands))
train['ord_5_band'] = train['ord_5_band'].replace(map_bands)

test['ord_5_band']=pd.cut(test.ord_5,pd.IntervalIndex(keys_bands))
test['ord_5_band'] = test['ord_5_band'].replace(map_bands)

test.ord_5_band.head()
train[nom_cols].nunique()
test[nom_cols].nunique()
for col in ["nom_7", "nom_8", "nom_9"]:

    train_vals = set(train[col].unique())

    test_vals = set(test[col].unique())

   

    ex=train_vals ^ test_vals

    if ex:

        train.loc[train[col].isin(ex), col]="x"

        test.loc[test[col].isin(ex), col]="x"
train[nom_cols].nunique()
test[nom_cols].nunique()
train[train.nom_7=='x']
train=train[train.nom_7!='x']
train[train.nom_7=='x']
labelEnc=LabelEncoder()
for col in nom_cols:

    train[col]=labelEnc.fit_transform(train[col])

    test[col]=labelEnc.fit_transform(test[col])

train[nom_cols].head()
train[bin_cols].head()
for col in ['bin_3', 'bin_4']:

    train[col]=labelEnc.fit_transform(train[col])

    test[col]=labelEnc.fit_transform(test[col])

test[bin_cols].head()
X_temp=train.drop('target', axis=1)

Y=train.target
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=4)

gm.fit(X_temp)
X_temp['Gaussian_Mixture']=gm.predict(X_temp)
test['Gaussian_Mixture']=gm.predict(test)
X_oh_temp=pd.get_dummies(X_temp.drop('id', axis=1), columns=X_temp.drop('id', axis=1).columns, drop_first=True, sparse=True)
X_oh=scipy.sparse.csr_matrix(X_oh_temp.values)
X_oh
test_oh_temp=pd.get_dummies(test.drop('id', axis=1), columns=test.drop('id', axis=1).columns, drop_first=True, sparse=True)
test_oh_temp.shape
test_oh=scipy.sparse.csr_matrix(test_oh_temp.values)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_oh, 

                                                      Y, 

                                                      test_size = 0.20,

                                                      random_state=42)

lr=LogisticRegression(C=0.125, solver="lbfgs", max_iter=500)  



lr.fit(X_train, Y_train)



y_pred_lr=lr.predict_proba(X_valid)
roc_auc_score(Y_valid.values, y_pred_lr[:,1])
lr.fit(X_oh, Y)
y_pred=lr.predict_proba(X_oh)

roc_auc_score(Y.values, y_pred[:,1])
test_pred_lr=lr.predict_proba(test_oh)
output_dict = {'id': test.id,

                       'target': test_pred_lr[:,1]}





output = pd.DataFrame(output_dict, columns = ['id', 'target'])

output.head(10)
output.to_csv('submission.csv', index=False)