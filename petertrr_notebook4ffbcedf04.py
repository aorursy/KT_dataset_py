import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df_store = pd.read_csv('../input/store.csv', index_col='Store')
df_store.head()
df_store.info()
categorial_features = ['StoreType', 'Assortment', 'PromoInterval']
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



for p in categorial_features:

    X_int = LabelEncoder().fit_transform(df_store[p].values.astype(str)).reshape(-1,1)

    ohe_feat = OneHotEncoder(sparse=False).fit_transform(X_int)

    tmp = pd.DataFrame(ohe_feat, columns=['{0}='.format(p) + str(i) for i in df_store[p].unique()], 

                       index=df_store.index,

                       dtype=int)

    df_store = pd.concat([df_store, tmp], axis=1)

    df_store = df_store.drop(p, axis=1)
for p in ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:

    df_store.loc[:, p] = (df_store[p] - df_store[p].mean()) / df_store[p].std()
df_store.head()
from sklearn.manifold import TSNE



model = TSNE()

arr = model.fit_transform(df_store.fillna(0))

plt.scatter(arr[:, 0], arr[:, 1])
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score



scores = []

ns = list(range(2, 10)) + list(range(10, 30, 5))

for n in ns:

    agc = AgglomerativeClustering(n_clusters=n)

    store_cluster = agc.fit_predict(df_store.fillna(0)).reshape(-1,1)

    scores.append(silhouette_score(df_store.fillna(0), store_cluster.ravel()))

plt.plot(ns, scores)
agc = AgglomerativeClustering(n_clusters=9)

store_cluster = agc.fit_predict(df_store.fillna(1))

store_cluster.shape
from sklearn.manifold import TSNE



model = TSNE()

arr = model.fit_transform(df_store.fillna(1))

plt.scatter(arr[:, 0], arr[:, 1], c=store_cluster)
df_train = pd.read_csv('../input/train.csv', index_col='Store')
print(df_train.shape)

df_train.head()
df_train.info()
df_train.StateHoliday.unique()
df_train.replace({'StateHoliday': {0: '0'}}, inplace=True)

df_train.StateHoliday.unique()
SH_int = LabelEncoder().fit_transform(df_train.StateHoliday.values.astype(str)).reshape(-1,1)

ohe_feat = OneHotEncoder(sparse=False).fit_transform(SH_int)

tmp = pd.DataFrame(ohe_feat, columns=['SH='+ str(i) for i in df_train.StateHoliday.unique()], 

                   index=df_train.index,

                   dtype=int)

df_train = df_train.drop('StateHoliday', axis=1)

df_train = pd.concat([df_train, tmp], axis=1)
ohe_feat = OneHotEncoder(sparse=False).fit_transform(df_train.DayOfWeek.values.reshape(-1,1))

tmp = pd.DataFrame(ohe_feat, columns=['DayOfWeek=' + str(i) for i in df_train.DayOfWeek.unique()], 

                   index=df_train.index,

                   dtype=int)

df_train = df_train.drop('DayOfWeek', axis=1)

df_train = pd.concat([df_train, tmp], axis=1)
print(df_train.shape, df_train.columns)
df_train['label'] = pd.Series([store_cluster[ind - 1] for ind in df_train.index],

                              index=df_train.index)



y_train = df_train[df_train.Open != 0].Sales.values

X_train = df_train[df_train.Open != 0].drop(['Date', 'Sales', 'Customers'], axis=1).values
from sklearn.linear_model import Ridge

from sklearn.metrics import accuracy_score



rlnrs = {}

for i, c in enumerate(np.unique(store_cluster)):

    df_c = df_train[df_train.label == c]

    if df_c.shape[0] == 0:

        continue

    X_c = df_c.drop(['Date', 'Sales', 'Customers'], axis=1).values

    y_c = df_c.Sales.values

    rlnr = Ridge()

    rlnr.fit(X_c, y_c)

    rlnrs.update({c: rlnr})

    print(c, rlnr.score(X_c, y_c))
df_test = pd.read_csv('../input/test.csv', index_col='Id')
print(df_test.shape)

df_test.head()
print(df_test.Open.unique())
df_test.loc[df_test.Open == 0, 'Sales'] = 0

df_test.Open = df_test.loc[:, 'Open'].fillna(1)
df_test.replace({'StateHoliday': {0: '0'}}, inplace=True)



SH_int = LabelEncoder().fit_transform(df_test.StateHoliday.values.astype(str)).reshape(-1,1)

ohe_feat = OneHotEncoder(sparse=False).fit_transform(SH_int)

tmp = pd.DataFrame(ohe_feat, columns=['SH='+ str(i) for i in df_test.StateHoliday.unique()], 

                   index=df_test.index,

                   dtype=int)

df_test = df_test.drop('StateHoliday', axis=1)

df_test = pd.concat([df_test, tmp], axis=1)

df_test['SH=b'] = 0

df_test['SH=c'] = 0



ohe_feat = OneHotEncoder(sparse=False).fit_transform(df_test.DayOfWeek.values.astype(str).reshape(-1,1))

tmp = pd.DataFrame(ohe_feat, columns=['DayOfWeek=' + str(i) for i in df_test.DayOfWeek.unique()], 

                   index=df_test.index,

                   dtype=int)

df_test = df_test.drop('DayOfWeek', axis=1)

df_test = pd.concat([df_test, tmp], axis=1)

    

df_test['label'] = pd.Series([store_cluster[ind - 1] for ind in df_test.Store],

                              index=df_test.index)



df_test = df_test.fillna(0)



X_test = df_test.drop(['Store', 'Date'], axis=1).values
print(df_test.shape, df_test.columns)
for c in rlnrs.keys():

    X_c = df_test[(df_test.label == c) & (df_test.Open != 0)].drop(['Store', 'Date', 'Sales'], axis=1).values

    df_test.loc[(df_test.label == c) & (df_test.Open != 0), 'Sales'] = rlnrs[c].predict(X_c)

df_test.Sales.mean(), df_test.Sales.min(), df_test.Sales.max()
out = pd.DataFrame({

    "Id": df_test.index,

    "Sales": df_test.Sales.values

})

out.to_csv('submission.csv', index=False)