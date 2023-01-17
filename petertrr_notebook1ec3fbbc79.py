import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df_store = pd.read_csv('../input/store.csv', index_col=None)
df_store.head()
df_store.info()
df_store.head()
df_train = pd.read_csv('../input/train.csv', index_col=None, low_memory=False)
print(df_train.shape)

df_train.head()
df_train.info()
df_train.StateHoliday.unique()
df_train.replace({'StateHoliday': {0: '0'}}, inplace=True)

df_train.StateHoliday.unique()
df_train.DayOfWeek = df_train.DayOfWeek.astype(str)  # for dictvectorizer
df = df_train[df_train.Open != 0].merge(df_store, on='Store').fillna(1)

df.drop(['Store', 'Date', 'Customers'], axis=1, inplace=True)
df.shape, df.columns
from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import MinMaxScaler



dv = DictVectorizer()

mm = MinMaxScaler()



X_train = dv.fit_transform(df.drop(['Sales'],axis=1).fillna(0).to_dict('records'))



X_train = mm.fit_transform(X_train.toarray())
y_train = df.Sales.values
print(X_train.shape, y_train.shape)
from sklearn.ensemble import RandomForestRegressor



rgr = RandomForestRegressor(n_estimators=25, verbose=True, n_jobs=8)

rgr.fit(X_train, y_train)

print(rgr.score(X_train, y_train))
df_test = pd.read_csv('../input/test.csv', index_col=None)
print(df_test.shape)

df_test.head()
df_test.replace({'StateHoliday': {0: '0'}}, inplace=True)

df_test.StateHoliday.unique()
df_pred = df_test[df_test.Open != 0].merge(df_store, on='Store').fillna(1)

df_pred.drop(['Id', 'Store', 'Date'], axis=1, inplace=True)
df_pred.shape
X_test = dv.transform(df_pred.fillna(0).to_dict('records'))



X_test = mm.transform(X_test.toarray())
X_test.shape
rgr.predict(X_test)[:10]
df_test.loc[df_test.Open != 0,'Sales'] = rgr.predict(X_test)

df_test.loc[df_test.Open == 0, 'Sales'] = 0
df_test.head()
out = pd.DataFrame({

    "Id": df_test.Id,

    "Sales": df_test.Sales.values

})

out.to_csv('submission.csv', index=False)