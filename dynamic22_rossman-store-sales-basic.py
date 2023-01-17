import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

sns.set_style('whitegrid')

%matplotlib inline
df_train = pd.read_csv("../input/train.csv")

df_store = pd.read_csv("../input/store.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_store.head()
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))

sns.countplot(x = 'Open', hue = 'DayOfWeek', data = df_train,)
df_train['Year'] = df_train['Date'].apply(lambda x: int(x[:4]))

df_train.Year.head()
df_train['Month'] = df_train['Date'].apply(lambda x: int(x[5:7]))

df_train.Month.head()
average_monthly_sales = df_train.groupby('Month')["Sales"].mean()

fig = plt.subplots(1,1,sharex=True,figsize=(10,5))

average_monthly_sales.plot(legend=True,marker='o',title="Average Sales")
average_daily_sales = df_train.groupby('Date')["Sales"].mean()

fig = plt.subplots(1,1,sharex=True,figsize=(25,8))

average_daily_sales.plot(title="Average Daily Sales")
average_daily_visits = df_train.groupby('Date')["Customers"].mean()

fig = plt.subplots(1,1,sharex=True,figsize=(25,8))

average_daily_visits.plot(title="Average Daily Visits")
fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))



average_monthly_sales = df_train.groupby('Month')["Sales"].mean()



# plot average sales over time (year-month)

ax1 = average_monthly_sales.plot(legend = False, ax = axis1, marker = 'o', 

                                title = "Avg. Monthly Sales")



ax1.set_xticks(range(len(average_monthly_sales)))

ax1.set_xticklabels(average_monthly_sales.index.tolist(), rotation=90)



average_monthly_sales_change = df_train.groupby('Month')["Sales"].sum().pct_change()

# plot precent change for sales over time(year-month)

ax2 = average_monthly_sales_change.plot(legend = False, ax = axis2, marker = 'o', 

                                        colormap = "summer", title = "% Change Monthly Sales")
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x ='Month', y ='Sales', data = df_train, ax=axis1)

sns.barplot(x ='Month', y ='Customers', data = df_train, ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='DayOfWeek', y='Sales', data = df_train, order = [1,2,3,4,5,6,7], ax = axis1)

sns.barplot(x='DayOfWeek', y='Customers', data = df_train, order = [1,2,3,4,5,6,7], ax = axis2)
sns.factorplot(x ="Year", y ="Sales", hue ="Promo", data = df_train,

                   size = 6, kind ="box", palette ="muted")
df_train.StateHoliday.unique()
df_train['StateHoliday'] = df_train['StateHoliday'].replace(0, '0')

df_train.StateHoliday.unique()
sns.factorplot(x ="Year", y ="Sales", hue ="StateHoliday", data = df_train, 

               size = 6, kind ="bar", palette ="muted")
df_train["HolidayBin"] = df_train['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

df_train.HolidayBin.unique()
sns.factorplot(x ="Month", y ="Sales", hue ="HolidayBin", data = df_train, 

               size = 6, kind ="bar", palette ="muted")
sns.factorplot(x="DayOfWeek", y="Customers", hue="HolidayBin", col="Promo", data=df_train,

                   capsize=.2, palette="YlGnBu_d", size=6, aspect=.75)
df_train.SchoolHoliday.unique()
sns.factorplot(x="DayOfWeek", y="Customers", hue="SchoolHoliday", col="Promo", data=df_train,

                   capsize=.2, palette="YlGnBu_d", size=6, aspect=.75)
average_customers = df_train.groupby('Month')["Customers"].mean()

average_sales = df_train.groupby('Month')['Sales'].mean()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(average_sales.index, average_sales.values,ax=axis1)

sns.barplot(average_customers.index, average_customers.values,ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.distplot(df_train.Sales, color="m",ax = axis1)

sns.distplot(df_train.Customers, color="r",ax = axis2)
df_store.head()
df_store.Store.unique()
df_train.Store.unique()
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
average_storetype = df_stores_new.groupby('StoreType')['Sales', 'Customers'].mean()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(average_storetype.index, average_storetype['Sales'], ax=axis1)

sns.barplot(average_storetype.index, average_storetype['Customers'], ax=axis2)
average_assortment = df_stores_new.groupby('Assortment')['Sales', 'Customers'].mean()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(average_assortment.index, average_assortment['Sales'], ax=axis1)

sns.barplot(average_assortment.index, average_assortment['Customers'], ax=axis2)
stores_sales_corr = df_stores_new[['Customers', 'Sales', 'CompetitionDistance', 'Promo2']]
stores_sales_corr.corr()
sns.jointplot(df_stores_new.Sales, df_stores_new.CompetitionDistance, kind = 'scatter', size = 8)

#sns.jointplot(df_stores_new.Customers, df_stores_new.CompetitionDistance, kind = 'scatter', size = 10)
store_ids = [169]

df_select_stores = df_train[df_train.Store.isin(store_ids)]

df_select_stores['MonthYear'] = df_select_stores.Date.apply(lambda x: str(x)[:7])

df_select_stores.head()
average_store_sales = df_select_stores.groupby(['MonthYear'])['Sales', 'Customers'].mean()

average_store_sales.head()
average_store_sales = average_store_sales.reset_index()

average_store_sales.head()
ax = average_store_sales['Sales'].plot(legend=True, marker='o', figsize=(15,4))



start, end = ax.get_xlim()

labels = list(np.arange(start, end, 1))



ax.set_xticks(labels)

ax.set_xticklabels(average_store_sales.iloc[labels]['MonthYear'], rotation = 90)



# competitor begins

y = df_store["CompetitionOpenSinceYear"].loc[df_store["Store"]  == store_ids[0]].values[0]

m = df_store["CompetitionOpenSinceMonth"].loc[df_store["Store"] == store_ids[0]].values[0]



#

ax.axvline(x = ((y - 2013) * 12) + (m - 1), linewidth = 3, color = 'grey')
from scipy import stats
sns.jointplot(x="Sales", y="Customers", data=df_stores_avg, kind="hex",

              color='k',

              ratio=3);
sns.distplot(df_stores_avg.Sales, kde=False, fit=stats.norm);
df_test.head()
df_test.info()
#

df_test['Year'] = df_test['Date'].apply(lambda x: int(x[:4]))

df_test.Year.head()
df_test['Month'] = df_test['Date'].apply(lambda x: int(x[5:7]))

df_test.Month.unique()
df_test['MonthYear'] = df_test['Date'].apply(lambda x: str(x)[:7])

df_test.MonthYear.head()
df_test["HolidayBin"] = df_test.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})

df_test.HolidayBin.unique()
df_train.head()
df_train.columns
from sklearn.linear_model import LinearRegression
df_test = df_test.fillna(df_test.mean())

df_test.isnull().any()
closed_store_ids = df_test["Id"][df_test["Open"] == 0].values



# remove all rows(store,date) that were closed

df_test = df_test[df_test["Open"] != 0]
df_test = df_test.drop(['Date', 'MonthYear', 'StateHoliday'], axis=1)
train_stores = dict(list(df_train.groupby('Store')))

test_stores = dict(list(df_test.groupby('Store')))

submission = pd.Series()

scores = []



for i in test_stores:

    

    # current store

    store = train_stores[i]

    

    # define training and testing sets

    X_train = store.drop(["Date", "Sales", "Customers", "Store", "StateHoliday"],axis=1)

    Y_train = store["Sales"]

    

    X_test  = test_stores[i].copy()



    

    store_ids = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    

    # Linear Regression

    lreg = LinearRegression()

    lreg.fit(X_train, Y_train)

    

    Y_pred = lreg.predict(X_test)

    

    scores.append(lreg.score(X_train, Y_train))



    # Xgboost

    # params = {"objective": "reg:linear",  "max_depth": 10}

    # T_train_xgb = xgb.DMatrix(X_train, Y_train)

    # X_test_xgb  = xgb.DMatrix(X_test)

    # gbm = xgb.train(params, T_train_xgb, 100)

    # Y_pred = gbm.predict(X_test_xgb)

    

    # append predicted values of current store to submission

    submission = submission.append(pd.Series(Y_pred, index=store_ids))



# append rows(store,date) that were closed, and assign their sales value to 0

submission = submission.append(pd.Series(0, index=closed_store_ids))



# save to csv file

submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})

submission.to_csv('rossmann_submission.csv', index=False)
submission.head()
submission[submission['Id'] == 544]