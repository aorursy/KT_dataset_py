import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline
df_test = pd.read_csv('../input/test.csv', sep=',')

df_train = pd.read_csv('../input/train.csv', sep=',')

df_store=pd.read_csv('../input/store.csv',sep=',')
df_test.isnull().any()
df_train.isnull().any()
df_store.isnull().any()
df_test.fillna(1,inplace=True)
df_train.head()
df_test.head()
df_train["Year"]=df_train["Date"].apply(lambda x: int(x[:4]))

df_test["Year"]=df_train["Date"].apply(lambda x: int(x[:4]))
df_train["Month"]=df_train["Date"].apply(lambda x: int(x[5:7]))

df_test["Month"]=df_train["Date"].apply(lambda x: int(x[5:7]))
df_train.head()
df_train.StateHoliday.unique()
df_train.groupby("StateHoliday")["Sales"].mean().plot(kind="bar")
df_train["StateHoliday2"] = df_train["StateHoliday"].replace(0,"0").map({"0": 0, "a": 1, "b": 1, "c": 1})

df_test["StateHoliday2"]= df_test["StateHoliday"].replace(0,"0").map({"0": 0, "a": 1, "b": 1, "c": 1})
df_train["StateHoliday2"].unique()
df_train["SchoolHoliday"].unique()
df_test.head()
df_train.drop(["Date","StateHoliday"],axis=1,inplace=True)

df_test.drop(["Date","StateHoliday"],axis=1,inplace=True)

df_train.head()
df_test.head()
train_stores = dict(list(df_train.groupby('Store')))

test_stores = dict(list(df_test.groupby('Store')))

submission = pd.Series()
for i in test_stores:

    store = train_stores[i]

    

    X_train = store.drop(["Store","Sales", "Customers"],axis=1).values

    Y_train = store["Sales"].values

    

    store_ids = test_stores[i]["Id"]

    X_test  = test_stores[i].drop(["Id","Store"],axis=1).values

    

    model = RandomForestRegressor(n_estimators=25)

    model.n_jobs = 8

    model.fit(X_train,Y_train)

    Y_test=model.predict(X_test)

    

    submission = submission.append(pd.Series(Y_test, index=store_ids))



submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})

submission.to_csv('submission.csv', index=False)