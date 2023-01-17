import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import validation_curve

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_store = pd.read_csv('../input/store.csv')
y = df_train["Sales"].values
df_train['Year'] = df_train['Date'].apply(lambda x: int(x[0:4]))

df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train['Day'] = df_train['Date'].apply(lambda x: int(x[8:10]))

df_test['Year'] = df_test['Date'].apply(lambda x: int(x[0:4]))

df_test['Month'] = df_test['Date'].apply(lambda x: int(x[5:7]))

df_test['Day'] = df_test['Date'].apply(lambda x: int(x[8:10]))
df_store.CompetitionDistance.fillna(value=0, inplace=True)

df_test.Open.fillna(value=0, inplace=True)

df_train.StateHoliday[df_train["StateHoliday"] == 0] = "0"
print(df_train.shape)

print(df_test.shape)

print(df_store.shape)
df_train.head()
df_test.head()
df_store.head()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))



sns.barplot(x='Year', y='Sales', data=df_train, ax=axis1)

sns.barplot(x='Year', y='Customers', data=df_train, ax=axis2)
df_train.query('Open == 1')[['Sales', 'Customers']].hist(bins=100, figsize=(13,7));
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))

sns.barplot(x='Month', y='Sales', data=df_train, ax=axis1)

sns.barplot(x='Month', y='Customers', data=df_train, ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))

sns.barplot(x='DayOfWeek', y='Sales', data=df_train, ax=axis1)

sns.barplot(x='DayOfWeek', y='Customers', data=df_train, ax=axis2)
df_train[['Sales', 'Customers']].corr()
df_DayOfWeek = pd.get_dummies(df_train.DayOfWeek, prefix='DayOfWeek')

df_StateHoliday = pd.get_dummies(df_train.StateHoliday, prefix="StateHoliday_")

df_train = pd.concat([df_train, df_DayOfWeek, df_StateHoliday], axis=1)

del df_train["Date"]

del df_train["Day"]

del df_train["Customers"]

del df_train["DayOfWeek"]

del df_train["Sales"]

del df_train["StateHoliday"]
df_StoreType = pd.get_dummies(df_store.StoreType, prefix='StoreType_')

df_Assortment = pd.get_dummies(df_store.Assortment, prefix='Assortment_')

df_store = pd.concat([df_store, df_StoreType, df_Assortment], axis=1)

del df_store["StoreType"]

del df_store["Assortment"]

del df_store["PromoInterval"]
df = pd.merge(df_train, df_store, how='left', on=['Store'])
df.fillna(0, inplace=True)
X = df.values[:,1:]
parametrs = range(40, 241, 40)
scores, tst_scr = validation_curve(RandomForestRegressor(n_jobs = 4), X[:20000],\

               y[:20000], 'n_estimators', parametrs, cv=5, scoring='r2', verbose=2)
scores_mean = scores.mean(axis=1)

scores_std = scores.std(axis=1)

tst_scr_mean = tst_scr.mean(axis=1)

tst_scr_std = tst_scr.std(axis=1)

plt.plot(parametrs, tst_scr_mean)

plt.fill_between(parametrs, tst_scr_mean + tst_scr_std, tst_scr_mean - tst_scr_std, alpha=0.3)

plt.plot(parametrs, scores_mean)

plt.fill_between(parametrs, scores_mean + scores_std, scores_mean - scores_std, alpha=0.3)
df.shape
parametrs = range(3, 24)
scores, tst_scr = validation_curve(RandomForestRegressor(n_estimators=120, n_jobs = 4), X[:20000], \

                                   y[:20000], 'max_features', parametrs, cv=3, scoring='r2', verbose=2)
scores_mean = scores.mean(axis=1)

scores_std = scores.std(axis=1)

tst_scr_mean = tst_scr.mean(axis=1)

tst_scr_std = tst_scr.std(axis=1)

plt.plot(parametrs, tst_scr_mean)

plt.fill_between(parametrs, tst_scr_mean + tst_scr_std, tst_scr_mean - tst_scr_std, alpha=0.3)

plt.plot(parametrs, scores_mean)

plt.fill_between(parametrs, scores_mean + scores_std, scores_mean - scores_std, alpha=0.3)
parametrs = range(4, 61, 4)
scores, tst_scr = validation_curve(RandomForestRegressor(n_estimators=120, n_jobs = 4, max_features=16), X[:20000], \

                                   y[:20000], 'max_depth', parametrs, cv=3, scoring='r2', verbose=2)
scores_mean = scores.mean(axis=1)

scores_std = scores.std(axis=1)

tst_scr_mean = tst_scr.mean(axis=1)

tst_scr_std = tst_scr.std(axis=1)

plt.plot(parametrs, tst_scr_mean)

plt.fill_between(parametrs, tst_scr_mean + tst_scr_std, tst_scr_mean - tst_scr_std, alpha=0.3)

plt.plot(parametrs, scores_mean)

plt.fill_between(parametrs, scores_mean + scores_std, scores_mean - scores_std, alpha=0.3)
model = RandomForestRegressor(n_estimators=120, max_depth=20, max_features=16, n_jobs=4, verbose=2)
model.fit(X, y)
idx = model.feature_importances_.argsort()[::-1]
ax = sns.barplot(x=model.feature_importances_[idx], y=df.drop('Store', axis=1).columns[idx])
df_DayOfWeek = pd.get_dummies(df_test.DayOfWeek, prefix='DayOfWeek')

df_StateHoliday = pd.get_dummies(df_test.StateHoliday, prefix="StateHoliday_")

df_StateHoliday = pd.concat([df_StateHoliday, pd.DataFrame(columns=['StateHoliday__b', 'StateHoliday__c'])], axis = 1)
df_StateHoliday.fillna(0, inplace=True)
del df_test["Date"]

del df_test["Day"]

del df_test["DayOfWeek"]

del df_test["StateHoliday"]

df_test = pd.concat([df_test, df_DayOfWeek, df_StateHoliday], axis=1)

test_df = pd.merge(df_test, df_store, how='left', on=['Store'])

test_df.fillna(0, inplace=True)
y_test_pred = model.predict(test_df.values[:,2:])
submission = pd.DataFrame({ "Id": test_df.Id, "Sales": y_test_pred.reshape(-1.1)})
submission
submission.to_csv("rossman.csv",index=False)