import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import matplotlib.dates 

import datetime 

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")

df_store = pd.read_csv("../input/store.csv")

df_test = pd.read_csv("../input/test.csv")

df_train.head()
closed_store_ids = df_test["Id"][df_test["Open"] == 0].values

closed_store_ids
df_train['Year'] = df_train['Date'].apply(lambda x: int(x[:4]))

df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train['Day'] = df_train['Date'].apply(lambda x: int(x[8:]))
df_train['Customers']

df_train['Sales']

CaS = pd.DataFrame()

CaS['Customers'] = df_train['Customers']

CaS['Sales'] = df_train['Sales']

correlationMatrix = CaS.corr().abs()

plt.subplots(figsize=(13, 9))

sns.heatmap(correlationMatrix,annot=True)
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))

sns.countplot(x = 'Open', hue = 'DayOfWeek', data = df_train,)
for temp_year in range (2013,2016):

    df_train1_temp = df_train[df_train.Year == temp_year]

    average_daily_sales = df_train1_temp.groupby('Date')["Sales"].mean()

    fig = plt.subplots(1,1,sharex=True,figsize=(25,8))

    average_daily_sales.plot(title="Average Daily Sales")
average_monthly_sales = df_train.groupby('Month')["Sales"].mean()

fig = plt.subplots(1,1,sharex=True,figsize=(10,5))

average_monthly_sales.plot(legend=True,marker='o',title="Average Sales")
df_train.StateHoliday.unique()
df_train['StateHoliday'] = df_train['StateHoliday'].replace(0, '0')

df_train.StateHoliday.unique()
df_train["HolidayBin"] = df_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})
sns.factorplot(x ="Year", y ="Sales", hue ="Promo", data = df_train,

                   size = 5, kind ="box", palette ="muted")

sns.factorplot(x ="Year", y ="Sales", hue ="SchoolHoliday", data = df_train,

                   size = 5, kind ="box", palette ="muted")

sns.factorplot(x ="Year", y ="Sales", hue ="HolidayBin", data = df_train,

                   size = 5, kind ="box", palette ="muted")
sns.factorplot(x ="Year", y ="Sales", hue ="StateHoliday", data = df_train, 

               size = 6, kind ="bar", palette ="muted")
sns.factorplot(x ="Month", y ="Sales", hue ="HolidayBin", data = df_train, 

               size = 6, kind ="bar", palette ="muted")
sns.factorplot(x="DayOfWeek", y="Customers", hue="HolidayBin", col="Promo", data=df_train,

                   capsize=.2, palette="YlGnBu_d", size=6, aspect=.75)
sns.factorplot(x="DayOfWeek", y="Customers", hue="SchoolHoliday", col="Promo", data=df_train,

                   capsize=.2, palette="YlGnBu_d", size=6, aspect=.75)
sns.distplot(df_train.Sales)
df_store.head()
total_sales_customers =  df_train.groupby('Store')['Sales', 'Customers'].sum()

total_sales_customers.head()
df_total_sales_customers = pd.DataFrame({'Sales':  total_sales_customers['Sales'],

                                         'Customers': total_sales_customers['Customers']}, 

                                         index = total_sales_customers.index)



df_total_sales_customers = df_total_sales_customers.reset_index()

df_total_sales_customers.head()
avg_sales_customers =  df_train.groupby('Store')['Sales', 'Customers'].mean()

avg_sales_customers.head()
df_avg_sales_customers = pd.DataFrame({'Sales':  avg_sales_customers['Sales'],

                                         'Customers': avg_sales_customers['Customers']}, 

                                         index = avg_sales_customers.index)



df_avg_sales_customers = df_avg_sales_customers.reset_index()



df_stores_avg = df_avg_sales_customers.join(df_store.set_index('Store'), on='Store')

df_stores_avg.head()
df_stores_new = df_total_sales_customers.join(df_store.set_index('Store'), on='Store')

df_stores_new.head()
average_storetype = df_stores_new.groupby('StoreType')['Sales', 'Customers', 'CompetitionDistance'].mean()



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,4))

sns.barplot(average_storetype.index, average_storetype['Sales'], ax=axis1)

sns.barplot(average_storetype.index, average_storetype['Customers'], ax=axis2)

sns.barplot(average_storetype.index, average_storetype['CompetitionDistance'], ax=axis3)



average_storetype.index
average_assortment = df_stores_new.groupby('Assortment')['Sales', 'Customers'].mean()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(average_assortment.index, average_assortment['Sales'], ax=axis1)

sns.barplot(average_assortment.index, average_assortment['Customers'], ax=axis2)
df_test['Year'] = df_test['Date'].apply(lambda x: int(x[:4]))

df_test['Month'] = df_test['Date'].apply(lambda x: int(x[5:7]))

df_test['Day'] = df_test['Date'].apply(lambda x: int(x[8:]))

df_test["HolidayBin"] = df_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})

del df_test['Date']

del df_test['StateHoliday']

df_test.head()
del df_train['Date']

del df_train['StateHoliday']
#df_test = df_test[df_test["Open"] != 0]

#df_test[df_test['Store'] == 1].head()

a = list()

for i in df_test['Store']:

      a.append(float(df_store['CompetitionDistance'][df_store['Store'] == i]))

df_test['CompetitionDistance'] = a
a = list()

for i in df_train['Store']:

      a.append(float(df_store['CompetitionDistance'][df_store['Store'] == i]))

df_train['CompetitionDistance'] = a

df_train['CompetitionDistance'] = df_train['CompetitionDistance'].fillna(df_train['CompetitionDistance'].mean())
df_train['CompetitionDistance'] = np.log(df_train['CompetitionDistance'])

df_test['CompetitionDistance'] = np.log(df_test['CompetitionDistance'])
df_train.head()
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



train_stores = dict(list(df_train.groupby('Store')))

test_stores = dict(list(df_test.groupby('Store')))
best_list_max_depth = []

best_list_n_estimators = []



for i in test_stores:

    store = train_stores[i]

    X_train = store.drop(["Sales", "Store", "Customers"],axis=1)

    Y_train = store["Sales"]

    X_test  = test_stores[i].copy()



    

    store_ids = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    X_train = X_train.fillna(X_train.mean())

    X_test = X_test.fillna(X_train.mean())

    

    estimator = RandomForestRegressor(random_state=123, criterion = 'mse')

    params = {'n_estimators': range(5, 20), 'max_depth': range(5, 25)}

    grid = GridSearchCV(estimator, params).fit(X_train, Y_train)

    best_list_max_depth.append(grid.best_params_['max_depth'])

    best_list_n_estimators.append(grid.best_params_['n_estimators'])

    print ("score", grid.best_score_)

    print ("params", grid.best_params_)
res_max_depth = round(np.array(best_list_max_depth).mean())

res_n_estimators = round(np.array(best_list_n_estimators).mean())
best_max_depth = round(np.array(best_list_max_depth).mean())

best_n_estimators = round(np.array(best_list_n_estimators).mean())

print(best_max_depth)

print(best_n_estimators)


submission = pd.Series()

for i in  test_stores:

    store = train_stores[i]

    X_train = store.drop(["Sales", "Store", "Customers"],axis=1)

    Y_train = store["Sales"]

    X_test  = test_stores[i].copy()



    

    store_ids = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    X_train = X_train.fillna(X_train.mean())

    X_test = X_test.fillna(X_train.mean())

    

    rfr  = RandomForestRegressor(random_state=123, n_estimators = best_n_estimators,

                                 max_depth = best_max_depth, criterion = 'mse')

    rfr.fit(X_train, Y_train)

    Y_pred = rfr.predict(X_test)

    

    submission = submission.append(pd.Series(Y_pred, index=store_ids))

submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})

submission.to_csv('rossmann_submission_good.csv', index=False)

print (submission.shape)