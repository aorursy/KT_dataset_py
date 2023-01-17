import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



sns.set_style('whitegrid')

%matplotlib inline



#Загружаем данные

df_train = pd.read_csv("../input/train.csv")

df_store = pd.read_csv("../input/store.csv")

df_test = pd.read_csv("../input/test.csv")



#Распарсим дату на год и месяц

df_train['Year'] = df_train['Date'].apply(lambda x: int(x[:4]))

df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train.head()
cust_sales = pd.DataFrame()



cust_sales['Customers'] = df_train['Customers']

cust_sales['Sales']     = df_train['Sales']

correlation_matrix = cust_sales.corr().abs()



plt.subplots(figsize=(13, 9))

sns.heatmap(correlation_matrix,annot=True)
df_train["HolidayBin"] = df_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})



sns.factorplot(x ="Year", y ="Sales", hue ="Promo", data = df_train,

                   size = 4, kind ="box", palette ="muted")

sns.factorplot(x ="Year", y ="Sales", hue ="SchoolHoliday", data = df_train,

                   size = 4, kind ="box", palette ="muted")

sns.factorplot(x ="Year", y ="Sales", hue ="HolidayBin", data = df_train,

                   size = 4, kind ="box", palette ="muted")
#разные типы каникул

df_train['StateHoliday'] = df_train['StateHoliday'].replace(0, '0')

df_train["HolidayBin"]   = df_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})



sns.factorplot(x ="Year", y ="Sales", hue ="StateHoliday", data = df_train, 

               size = 6, kind ="bar", palette ="muted")
avg       = df_train.groupby('Month')["Customers"].mean()

avg_sales = df_train.groupby('Month')['Sales'].mean()

total_sales_customers =  df_train.groupby('Store')['Sales', 'Customers'].sum()

total_sales_customers.head()
df_total_sales_customers = pd.DataFrame({'Sales':  total_sales_customers['Sales'],

                                         'Customers': total_sales_customers['Customers']}, 

                                         index = total_sales_customers.index)



df_total_sales_customers = df_total_sales_customers.reset_index()

df_total_sales_customers.head()
avg_sales_customers =  df_train.groupby('Store')['Sales', 'Customers'].mean()

avg_sales_customers.head()
df_avg_sales_customers = pd.DataFrame({'Sales':     avg_sales_customers['Sales'],

                                       'Customers': avg_sales_customers['Customers']}, 

                                       index =      avg_sales_customers.index)



df_avg_sales_customers = df_avg_sales_customers.reset_index()



df_stores_avg = df_avg_sales_customers.join(df_store.set_index('Store'), on='Store')

df_stores_avg.head()
df_stores_new = df_total_sales_customers.join(df_store.set_index('Store'), on='Store')

df_stores_new.head()
#Больше всего посетителей и продаж в аптеках класса b

#Как это ни странно, но несмотря на то, что у магазинов класса b больше всего посетителей и продаж, 

#конкуренты находятся ближе всего к этому классу



average_storetype = df_stores_new.groupby('StoreType')['Sales', 'Customers', 'CompetitionDistance'].mean()



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,4))

sns.barplot(average_storetype.index, average_storetype['Sales'], ax=axis1)

sns.barplot(average_storetype.index, average_storetype['Customers'], ax=axis2)

sns.barplot(average_storetype.index, average_storetype['CompetitionDistance'], ax=axis3)
# В целом оказывается, что чем ближе конкурент, тем продажи ниже

comp_sales_cust = pd.DataFrame()



comp_sales_cust['Customers'] = average_storetype['Customers']

comp_sales_cust['Sales']     = average_storetype['Sales']

comp_sales_cust['Comp']      = average_storetype['CompetitionDistance']

corr_matrix = comp_sales_cust.corr()



plt.subplots(figsize=(13, 9))

sns.heatmap(corr_matrix,annot=True)
#Рассмотрим на ассортимент

avg_assort = df_stores_new.groupby('Assortment')['Sales', 'Customers'].mean()



fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(avg_assort.index, avg_assort['Sales'],     ax=axis1)

sns.barplot(avg_assort.index, avg_assort['Customers'], ax=axis2)
df_train = pd.read_csv("../input/train.csv")

df_store = pd.read_csv("../input/store.csv")

df_test  = pd.read_csv("../input/test.csv")
closed_store_ids = df_test["Id"][df_test["Open"] == 0].values



df_train['Year']       = df_train['Date'].apply(lambda x: int(x[:4]))

df_train['Month']      = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train["HolidayBin"] = df_train.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})



del df_train['Date']

del df_train['StateHoliday']



df_train.head()
df_test['Year']       = df_test['Date'].apply(lambda x: int(x[:4]))

df_test['Month']      = df_test['Date'].apply(lambda x: int(x[5:7]))

df_test["HolidayBin"] = df_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})



del df_test['Date']

del df_test['StateHoliday']



df_test.head()
df_test = df_test[df_test["Open"] != 0]

df_test[df_test['Store'] == 1].head()
# Переделать в numpy

a = list()

for i in df_test['Store']:

      a.append(float(df_store['CompetitionDistance'][df_store['Store'] == i]))



df_test['CompetitionDistance'] = a

df_test.head()
a = list()

for i in df_train['Store']:

      a.append(float(df_store['CompetitionDistance'][df_store['Store'] == i]))



df_train['CompetitionDistance'] = a

df_train['CompetitionDistance'] = df_train['CompetitionDistance'].fillna(df_train['CompetitionDistance'].mean())



df_train.head()
#Прологарифмируем данные, чтобы было совсем ништяк

df_train['CompetitionDistance'] = np.log(df_train['CompetitionDistance'])

df_test ['CompetitionDistance'] = np.log(df_test ['CompetitionDistance'])
#В качестве алгоритма регрессии был выбран Random Forest.

#Малое количество эстиматоров было выбрано по той причине, что сами датафреймы не очень большие 

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor



train_stores = dict(list(df_train.groupby('Store')))

test_stores = dict(list(df_test.groupby('Store')))

submission = pd.Series()



for i in test_stores:

    

    store = train_stores[i]

    X_train = store.drop(["Sales", "Store", "Customers"],axis=1)

    Y_train = store["Sales"]

    X_test  = test_stores[i].copy()



    store_ids = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    X_train = X_train.fillna(X_train.mean())

    X_test = X_test.fillna(X_train.mean())

    

    #RFR

    rfr = RandomForestRegressor(n_estimators = 5, criterion = 'mse')

    rfr.fit(X_train, Y_train)

    Y_pred = rfr.predict(X_test)

 

    submission = submission.append(pd.Series(Y_pred, index=store_ids))



submission = submission.append(pd.Series(0, index=closed_store_ids))

submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})



submission.to_csv('rossmann_submission.csv', index=False)