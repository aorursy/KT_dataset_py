import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12,8)



# Для кириллицы на графиках

font = {'family': 'Verdana',

        'weight': 'normal'}

plt.rc('font', **font)
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_store = pd.read_csv("../input/store.csv")
data_train.head()
data_train.shape
data_test.columns
data_store.head(n=3)
data_train.StateHoliday = data_train.StateHoliday.replace(0,'0')

data_test.StateHoliday = data_test.StateHoliday.replace(0,'0')

data_train.DayOfWeek.value_counts()
#fig, (axis1) = plt.subplots(1,1,figsize=(8,3))

#sns.countplot(x = 'Open', hue = 'DayOfWeek', data = data_train,)
data_train['Year'] = data_train['Date'].apply(lambda x: int(x[:4]))

data_train['Month'] = data_train['Date'].apply(lambda x: int(x[5:7]))

data_train.head()
average_sales_per_month = data_train.groupby('Month')["Sales"].mean()

plt.figure(figsize=(8, 5))

average_sales_per_month.plot(legend=True, marker='o', title="Average sales per month")
average_sales_per_day = data_train.groupby('Date')["Sales"].mean()

fig = plt.subplots(1,1, sharex=True, figsize=(18, 5))

average_sales_per_day.plot(legend=True, title="Average Daily Sales")
columns_corr = ['Sales', 'Customers', 'Promo', 'StateHoliday', 'SchoolHoliday']

data_train[columns_corr].corr(method='pearson')
#sns.factorplot(x ="Year", y ="Sales", hue ="Promo", data = data_train, size = 3, kind ="box", palette ="muted")
#sns.factorplot(x ="Year", y ="Sales", hue ="SchoolHoliday", data = data_train, size = 3, kind ="box", palette ="muted")
data_train['StateHoliday'] = data_train['StateHoliday'].replace(0, '0')

data_train["HolidayBin"] = data_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

data_train.StateHoliday.unique()
#sns.factorplot(x ="Year", y ="Sales", hue ="HolidayBin", data = data_train, size = 4, kind ="bar", palette ="muted")
average_customers_per_month = data_train.groupby('Month')['Customers'].mean()

average_sales_per_month = data_train.groupby('Month')['Sales'].mean()
plt.figure(figsize=(6, 4))

plt.plot(average_sales_per_month)
plt.figure(figsize=(6, 4))

plt.plot(average_customers_per_month)
total_customers_for_store =  data_train.groupby('Store')['Sales', 'Customers'].sum()
data_total_customers_for_store = pd.DataFrame({'Sales':  total_customers_for_store['Sales'],

                                               'Customers': total_customers_for_store['Customers']}, 

                                              index = total_customers_for_store.index)
data_total_customers_for_store = data_total_customers_for_store.reset_index()
average_sales_customers =  data_train.groupby('Store')['Sales', 'Customers'].mean()
data_average_sales_customers = pd.DataFrame({'Sales':  average_sales_customers['Sales'],

                                         'Customers': average_sales_customers['Customers']}, 

                                         index = average_sales_customers.index)



data_average_sales_customers = data_average_sales_customers.reset_index()



data_stores_average = data_average_sales_customers.join(data_store.set_index('Store'), on='Store')

data_stores_average.head()
data_average_sales_customers = pd.DataFrame({'Sales':  average_sales_customers['Sales'],

                                         'Customers': average_sales_customers['Customers']}, 

                                         index = average_sales_customers.index)
data_average_sales_customers = data_average_sales_customers.reset_index()
data_stores_average = data_average_sales_customers.join(data_store.set_index('Store'), on='Store')
data_stores_average.head(n=3)
data_stores_new = data_total_customers_for_store.join(data_store.set_index('Store'), on='Store')
average_store_type = data_stores_new.groupby('StoreType')['Sales', 'Customers', 'CompetitionDistance'].mean()
#fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(12,3))

#sns.barplot(average_store_type.index, average_store_type['Sales'], ax=axis1)

#sns.barplot(average_store_type.index, average_store_type['Customers'], ax=axis2)

#sns.barplot(average_store_type.index, average_store_type['CompetitionDistance'], ax=axis3)
Data_cmp = pd.DataFrame()

Data_cmp['Customers'] = average_store_type['Sales']

Data_cmp['Sales'] = average_store_type['Customers']

Data_cmp['Comp'] = average_store_type['CompetitionDistance']
columns_corr = ['Sales', 'Customers', 'Comp']

Data_cmp[columns_corr].corr(method='pearson')
average_assort = data_stores_new.groupby('Assortment')['Sales', 'Customers'].mean()
#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,3))

#sns.barplot(average_assort.index, average_assort['Sales'], ax=axis1)

#ns.barplot(average_assort.index, average_assort['Customers'], ax=axis2)
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_store = pd.read_csv("../input/store.csv")
closed_store_data = data_test["Id"][data_test["Open"] == 0].values

data_train.StateHoliday = data_train.StateHoliday.replace(0,'0')

data_test.StateHoliday = data_test.StateHoliday.replace(0,'0')
data_train['Year'] = data_train['Date'].apply(lambda x: int(x[:4]))

data_train['Month'] = data_train['Date'].apply(lambda x: int(x[5:7]))

data_train["HolidayBin"] = data_train.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
del data_train['Date']

del data_train['StateHoliday']
data_test['Year'] = data_test['Date'].apply(lambda x: int(x[:4]))

data_test['Month'] = data_test['Date'].apply(lambda x: int(x[5:7]))

data_test["HolidayBin"] = data_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
del data_test['Date']

del data_test['StateHoliday']
data_test = data_test[data_test["Open"] != 0]
data_test[data_test['Store'] == 1].head()
arr_tmp = []

for i in data_test['Store']:

      arr_tmp.append(float(data_store['CompetitionDistance'][data_store['Store'] == i]))

data_test['CompetitionDistance'] = arr_tmp
arr_tmp = []

for i in data_train['Store']:

      arr_tmp.append(float(data_store['CompetitionDistance'][data_store['Store'] == i]))

data_train['CompetitionDistance'] = arr_tmp

data_train['CompetitionDistance'] = data_train['CompetitionDistance'].fillna(data_train['CompetitionDistance'].mean())
train_stores = dict(list(data_train.groupby('Store')))

test_stores = dict(list(data_test.groupby('Store')))
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
# Выберем лучшую модель (аналогично для линейной регрессии)

# res_list_depth = []

# res_list_nestim = []

# for i in range (1, 1000, 50):

#     store_part = train_stores[i]

#     X_train_part = store_part.drop(["Sales", "Store", "Customers"],axis=1)

#     Y_train_part = store_part["Sales"]

#     X_train_part = X_train_part.fillna(X_train_part.mean())

#     estimator = RandomForestRegressor(random_state=42, criterion = 'mse')

#     params = {'n_estimators': range(3, 15), 'max_depth': range(3, 30)}

#     grid = GridSearchCV(estimator, params).fit(X_train_part, Y_train_part)

#     res_list_depth.append(grid.best_params_['max_depth'])

#     res_list_nestim.append(grid.best_params_['n_estimators'])

#     print i

#     print "score", grid.best_score_

#     print "params", grid.best_params_

# print np.array(res_list_depth).mean()

# print np.array(res_list_nestim).mean() 
# Возьмем модель с лучшими параматрами

result = pd.Series()

for i in test_stores:  

    store = train_stores[i]

    X_train = store.drop(["Sales", "Store", "Customers"],axis=1)

    Y_train = store["Sales"]

    X_test  = test_stores[i].copy()   

    store_ind = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    X_train = X_train.fillna(X_train.mean())

    X_test = X_test.fillna(X_train.mean())

    estimator = RandomForestRegressor(n_estimators=10, max_depth=13, criterion = 'mse')

    estimator.fit(X_train, Y_train)

    Y_pred = estimator.predict(X_test)

    result = result.append(pd.Series(Y_pred, index=store_ind))

result = result.append(pd.Series(0, index=closed_store_data))

result = pd.DataFrame({ "Id": result.index, "Sales": result.values})

result.to_csv('result_new.csv', index=False)
# Лучший результат до дедлайна: 0.14930 - pub, 0.13491 - priv