import pandas as pd

import numpy as np

#from googletrans import Translator



from itertools import product

import itertools



import matplotlib.pyplot as plt

import statsmodels.api as sm

import matplotlib

import seaborn as sns



from sklearn.metrics import mean_squared_error

from sklearn import metrics

from statsmodels.tsa.stattools import adfuller,pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.gofplots import qqplot

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf



from pylab import rcParams

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'



import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs

from pandas.plotting import autocorrelation_plot



from statsmodels.tsa.seasonal import seasonal_decompose

from scipy import stats
"""

import re

translator = Translator()

translate = ["item_name","item_category_name","shop_name"]

shops = pd.read_csv("shops.csv")

shops_lst = list(shops.shop_name.unique())

shops["shop_name_en"] = shops["shop_name"].apply(translator.translate, src = "ru", dest = "en").apply(getattr, args=('text',))



shops = shops.drop(columns = {"shop_name"})

shop_lst = list(shops.shop_name_en)

list_of_list_shops = [re.findall(r'[a-zA-Z]+', i) for i in shop_lst]

shops["City"] = [list_of_list_shops[i][0] +" "+ list_of_list_shops[i][1] 

                 if ((list_of_list_shops[i][0] == "St") |(list_of_list_shops[i][0] == "Itinerant") | 

                                                                             (list_of_list_shops[i][0] =="Digital"))

                 else list_of_list_shops[i][0] + " "+ list_of_list_shops[i][1] +" "+ list_of_list_shops[i][2] 

                 if (list_of_list_shops[i][0] == "Shop")

                 else list_of_list_shops[i][0] for i in range(len(list_of_list_shops))]



shops.to_csv("shops_new.csv", sep = ";")

"""
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

item_categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
sales_train.date = pd.to_datetime(sales_train.date)
sales_train_after11 = sales_train[(sales_train["date"] >= "2015-11-01")]

sales_train_after11.date.unique()
sales_train = sales_train[(sales_train["date"] < "2015-11-01")]

sales_train.shape
sales_train.head()
items = pd.merge(items, item_categories, on = "item_category_id")

items.shape
sales_train = pd.merge(sales_train, items, on = "item_id")

sales_train.head()
sales_train = pd.merge(sales_train, shops, on = "shop_id")

sales_train.head()
sales_train.shape
sales_train = sales_train[(sales_train["date"] < "2015-11-01")]

sales_train.shape
sales_per_product = sales_train.groupby("item_name", as_index=False).agg({"item_cnt_day":"sum"}).sort_values(by = "item_cnt_day", ascending = False)[0:10]

sales_per_product
ax = sns.barplot(x = "item_cnt_day", y = "item_name", data = sales_per_product)

plt.figure(figsize=(20,10))

plt.tight_layout()

#sns.set_style("whitegrid")

ax.set_title("Bestseller",y= 1.1, fontsize=18, weight = "semibold")

ax.set_xlabel("# of products", fontsize = 18, weight = "semibold")

ax.set_ylabel("Products", fontsize = 18, weight = "semibold")

sales_per_shop = sales_train.groupby(by = "shop_name", as_index=False).agg({"item_cnt_day":"sum"}).sort_values(by = "item_cnt_day",ascending = False)[0:10]
ax = sns.barplot(x = "item_cnt_day", y = "shop_name", data = sales_per_shop, palette="gist_heat")

sns.set_style("whitegrid")



ax.set_title("Shops sort by amount of sold products",y= 1.1, fontsize=20, weight = "semibold")

ax.set_xlabel("Amount of products", fontsize = 18, weight = "semibold")

ax.set_ylabel("shops", fontsize = 18, weight = "semibold")

sales_train["revenue"] = sales_train["item_cnt_day"]*sales_train["item_price"]
sales_train.date.min(), sales_train.date.max()
sales = sales_train.groupby('date')['item_cnt_day'].sum().reset_index()
sales = sales.set_index('date')
sales.index
sales.dtypes
y = sales['item_cnt_day'].resample('MS').mean()
y["2015":]
# plot historical data about all sold products per day

y.plot(figsize=(15, 6))

plt.show()
coefficients, residuals, _, _, _ = np.polyfit(range(len(y.index)),y,1,full=True)

mse = residuals[0]/(len(y.index))

nrmse = np.sqrt(mse)/(y.max() - y.min())

print('Slope ' + str(coefficients[0]))

print('NRMSE: ' + str(nrmse))
(y[33]-y[0])/y[0]
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, freq=12, model='additive')

fig = decomposition.plot()

plt.show()
# check the sum of item_cnt_day per day.

result = adfuller(y)

print("Daily Basis:")

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))

    

# p-value is smaller than 0.05 so we can reject the Null Hypothesis, 

# the time series is stationary and has no time dependent structure
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
# The best AIC is:

# ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:115.62002802642752



mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2013':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('date')

ax.set_ylabel('item_cnt_day')

plt.legend()

plt.show()
y_forecasted = pred.predicted_mean

y_truth = y['2015-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('date')

ax.set_ylabel('item_cnt_day')

plt.legend()

plt.show()
pred_uc = results.get_forecast(steps=3)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('date')

ax.set_ylabel('item_cnt_day')

plt.legend()

plt.show()
pred_uc = results.get_forecast(steps=24)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date',fontsize=18, weight = "bold")

ax.set_ylabel('Amount of sold items',fontsize=18, weight = "semibold")

plt.title("Forecast sold products next 2 years",y= 1.1, fontsize=18, weight = "semibold" )

plt.legend()

plt.show()

shops_lst = list(sales_train.shop_id.unique())
online_shop = sales_train[(sales_train["shop_id"] == 12)]
offline_shop = sales_train[(sales_train["shop_id"] != 12)]
offline_sales = offline_shop.groupby(['date',"shop_id"])['item_cnt_day'].sum().reset_index()
offline_sales = offline_sales.set_index('date')
offline_sales.dtypes
# plot time series for each offline shop

for i in shops_lst:

    sales_shop = offline_sales[(offline_sales["shop_id"] == i)]

    y = sales_shop["item_cnt_day"].resample("MS").mean()

    

    y.plot()

   

    

sales_per_online_shop = online_shop.groupby(['date',"shop_id"])['item_cnt_day'].sum().reset_index()
sales_per_online_shop = sales_per_online_shop.set_index('date')
o = sales_per_online_shop["item_cnt_day"].resample("MS").mean()

#X = sales_per_online_shop.index



#plt.plot(X, coefficients[0]*X +residuals, color="red")

axo = o.plot()



plt.title("Online Sales", y= 1.1, fontsize=18, weight = "semibold")

plt.xlabel("Date", fontsize=14, weight = "semibold")

plt.ylabel("# sold products", fontsize=14, weight = "semibold")

plt.show()

coefficients, residuals, _, _, _ = np.polyfit(range(len(o.index)),o,1,full=True)

mse = residuals[0]/(len(o.index))

nrmse = np.sqrt(mse)/(o.max() - o.min())

print('Slope ' + str(coefficients[0]))

print('NRMSE: ' + str(nrmse))

(o[33]-o[0])/o[0]*100
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(o, freq=12, model='additive')

fig = decomposition.plot()

plt.show()
# check the sum of item_cnt_day per day.

result = adfuller(o)

print("Daily Basis:")

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))

    

# p-value is smaller than 0.05 so we can reject the Null Hypothesis, 

# the time series is stationary and has no time dependent structure
p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(o,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
# The best AIC is:

# ARIMA(1, 1, 0)x(1, 1, 0, 12)12 - AIC:87.79639573691884



mod = sm.tsa.statespace.SARIMAX(o,

                                order=(1, 1, 0),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = o['2013':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('date')

ax.set_ylabel('item_cnt_day')

plt.legend()

plt.show()
o_forecasted = pred.predicted_mean

o_truth = o['2015-01-01':]

mse = ((o_forecasted - o_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

ax = o.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('date')

ax.set_ylabel('item_cnt_day')

plt.legend()

plt.show()
pred_uc = results.get_forecast(steps=15)

pred_ci = pred_uc.conf_int()

ax = o.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('date')

ax.set_ylabel('item_cnt_day')

plt.legend()

plt.show()
last_offline_sales = sales_train[(sales_train.date_block_num == 33) & (sales_train.shop_id != 12)]

w = last_offline_sales.item_cnt_day.sum()
last_online_sales = sales_train[(sales_train.date_block_num == 33) & (sales_train.shop_id == 12)]

z = last_online_sales.item_cnt_day.sum()
labels = ['Offline', "online"]

sizes = [(w/(w+z)),(z/(w+z))]
explode = (0, 0.1)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',  startangle=60)

plt.axis('equal', fontsize=14, weight = "semibold")



plt.show()
