import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

from collections import Counter



from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE

import os

from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor
def load_year(y):

    dtypes = pickle.load(open(

        f'../input/kdd2020-cpr/{y}_dtypes.pkl', 'rb'

    ))

    del dtypes['date']

    df = pd.read_csv(

        f'../input/kdd2020-cpr/{y}.csv',

        dtype=dtypes, parse_dates=['date']

    )

    return df
YEAR = 2018

COMPONENTS = 110

XGB_ESTIMATORS = 60

MAGICIAN = 'TruncatedSVD'
%%time

df = load_year(YEAR)
pd.Series(df.dtypes).value_counts()
%%time

def drop_fullnull(df, inplace=False):

    mask = df.isnull().all()

    labels = df.columns[mask]

    

    shape = df.shape

    print(labels)

    if inplace:

        df.drop(labels=labels, axis=1, inplace=True)

    else:

        df = df.drop(labels=labels, axis=1)

    if labels.any():

        if shape == df.shape:

            print('lables:', labels)

        else:

            print(shape, df.shape)

    return df



clean = df

# clean = drop_fullnull(df)

clean.shape
vclean = clean.copy()

vclean.fillna(0, inplace=True)
%%time

magician = TruncatedSVD(n_components=COMPONENTS, random_state=0)
%%time

cp = df.copy()



cp.fillna(0, inplace=True)



out_mask = df.columns.str.contains('output')

out_cols = df.columns[df.columns.str.contains('output')]

cp_out = cp.loc[:, out_mask]

cp.drop(['id'] + out_cols.tolist(),

        axis=1, inplace=True

)

cp['day'] = cp.date.dt.day

cp['month'] = cp.date.dt.month

cp['year'] = cp.date.dt.year

cp.drop('date', inplace=True, axis=1)



magician.fit(cp)
out_cols.ravel()[[0,4,10,-1]]
%%time

model = XGBRegressor(

    n_estimators=XGB_ESTIMATORS, random_state=0, n_jobs=-1,

    learning_rate=.1, max_depth=10, tree_method='gpu_hist', verbosity=2,

)

clf = MultiOutputRegressor(model)



clf.fit(pd.DataFrame(magician.transform(cp)), cp_out);
%%time

d9 = load_year(2019)

d9.shape
%%time



def svd_preprocess(df):

    cp = df.copy()

    ids = df.id.copy()

    

    cp.fillna(0, inplace=True)

    cp.drop(['id'], axis=1, inplace=True, errors='ignore')

    cp['day'] = cp.date.dt.day

    cp['month'] = cp.date.dt.month

    cp['year'] = cp.date.dt.year

    cp.drop('date', inplace=True, axis=1)

    return cp, ids



X, ids = svd_preprocess(d9)



df_in = pd.DataFrame(magician.transform(X))

df_in.index = ids

df_in.head(1)
y_pred = clf.predict(df_in)



out_cols_flat = out_cols.ravel()

id_col = []

for i in ids:

    id_col.extend([f'{i}_{sufix}' for sufix in out_cols_flat])
df_sub = pd.DataFrame(

    {'id':id_col , 'value':y_pred.ravel()}

)

df_sub.to_csv(

    f'submission-{YEAR}-{COMPONENTS}{MAGICIAN}-{XGB_ESTIMATORS}xgb.csv',

    index=False

)

pickle.dump(clf, open('clf.pkl', 'wb'))