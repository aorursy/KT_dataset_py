#Import packages /libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from math import sqrt

#Modeling
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from fbprophet import Prophet

#Evaluation
from sklearn.metrics import mean_squared_error
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric

#Menghilangkan warning
import warnings
warnings.filterwarnings("ignore")
#Melihat data yang ada di directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Load semua data
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv') #ini kita tidak pakai, karena hanya untuk input kompetisi
category = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
#Mengabungkan semua data
data_1 = pd.merge(items, sales, on='item_id', how='inner')
data_2 = pd.merge(data_1, shops, on='shop_id', how='inner')
data = pd.merge(data_2, category, on='item_category_id', how='inner')
#Melihat sekilas data yang telah digabung
data.head()
#Melihat info pada data 
data.info()
#Merubah tipe data kolom item_cnt_day menjadi integer, karena inputnya adalah berapa barang yang terjual
data['item_cnt_day'] = data.item_cnt_day.astype('int')
#Merubah kolom date menjadi index dan merubah tipe datanya menjadi date
data['date'] = pd.to_datetime(data.date)
data =  data.sort_values('date').reset_index(drop=True)
#Melihat kembali dataset kita apakah kolom date sudah sesuai
data.head()
#Melihat apakah ada input yang kosong pada data kita
data.isnull().sum() /len(data) *100
#Membuat kolom total_sales untuk melihat revenue / pendapatan perhari
data['total_sales'] = data['item_price'] * data['item_cnt_day']
data.head()
#Kita akan menggunakan Inter Quartile Range untuk menangani outliers
#Menentukan Limit
def limit(i):
    Q1,Q3 = np.percentile(data[i] , [25,75])
    IQR = Q3 - Q1
    
    #menentukan upper limit biasa dan upper limit ekstim
    lower_limit = Q1 - (IQR * 1.5)
    lower_limit_extreme = Q1 - (IQR * 3)
    upper_limit = Q3 + (IQR * 1.5)
    upper_limit_extreme = Q3 + (IQR * 3)
    print('Lower Limit:', lower_limit)
    print('Lower Limit Extreme:', lower_limit_extreme)
    print('Upper Limit:', upper_limit)
    print('Upper Limit Extreme:', upper_limit_extreme)

#Mengitung persen outliers dari data    
def percent_outliers(i):
    Q1,Q3 = np.percentile(data[i] , [25,75])
    IQR = Q3 - Q1
    
    #menentukan upper limit biasa dan upper limit ekstim
    lower_limit = Q1 - (IQR * 1.5)
    lower_limit_extreme = Q1 - (IQR * 3)
    upper_limit = Q3 + (IQR * 1.5)
    upper_limit_extreme = Q3 + (IQR * 3)
    #melihat persenan outliers terhadap total data
    print('Lower Limit: {} %'.format(data[(data[i] >= lower_limit)].shape[0]/ data.shape[0]*100))
    print('Lower Limit Extereme: {} %'.format(data[(data[i] >= lower_limit_extreme)].shape[0]/data.shape[0]*100))
    print('Upper Limit: {} %'.format(data[(data[i] >= upper_limit)].shape[0]/ data.shape[0]*100))
    print('Upper Limit Extereme: {} %'.format(data[(data[i] >= upper_limit_extreme)].shape[0]/data.shape[0]*100))
#Melihat outliers pada kolom item_price
sns.boxplot(x=data["item_price"])
#Melihat apakah ada harga item yang 0 atau minus. Karena jika ada ini merupakan sebuah kesalahan
data[data['item_price'] <= 0].count()
#Menghilangkan input yang <= 0
data = data[data['item_price'] > 0]
#Melihat IQR dari kolom item_price
print(limit('item_price'))
print('-'*50)
print(percent_outliers('item_price'))
#Melihat outliers pada kolom item_cnt_day
sns.boxplot(x=data["item_cnt_day"])
#Melihat IQR pada kolom itm_cnt_day
print(limit('item_cnt_day'))
print('-'*50)
print(percent_outliers('item_cnt_day'))
#Mengecek apakah ada item yang dijual dalam jumlah 0 atau kurangdari 0, karena jika ada itu merupakan kesalahan
#Karena tida kmungkin item dijual dengan jumlah 0 (karena kita hanya mengambil yang laku) atau bahkan minus
data[data['item_cnt_day'] <= 0].count()
#Menghilangkan input yang <= 0
data = data[data['item_cnt_day'] > 0]
#Produk apa yang paling laris?
top_10_product_best_seller = data.groupby(['item_name'])['item_cnt_day'].sum().sort_values(ascending=False)[:10]

#Visualisasi 
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_product_best_seller.index,x=top_10_product_best_seller.values)
plt.title('Top 10 Most Selling Items',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Item Name',fontsize=17)
top_10_product_best_seller
#Produk apa yang paling tidak laris?
top_10_product_least_seller = data.groupby(['item_name'])['item_cnt_day'].sum().sort_values(ascending=True)[:10]

#Visualisasi 
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_product_least_seller.index,x=top_10_product_least_seller.values)
plt.title('Top 10 Least Selling Items',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Item Name',fontsize=17)
#Toko apa yang paling laris?
top_10_seller = data.groupby(['shop_name'])['item_cnt_day'].sum().sort_values(ascending=False)[:10]

#Visualisasi 
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_seller.index,x=top_10_seller.values)
plt.title('Top 10 Most Selling Shop',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Shop Name',fontsize=17)
top_10_seller
#Toko apa yang paling tidak laris?
top_10_least_seller = data.groupby(['shop_name'])['item_cnt_day'].sum().sort_values(ascending=True)[:10]

#Visualisasi 
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_least_seller.index,x=top_10_least_seller.values)
plt.title('Top 10 Least Shop',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Shop Name',fontsize=17)
top_10_least_seller
#Toko apa yang paling laris?
top_10_seller = data.groupby(['shop_name'])['total_sales'].sum().sort_values(ascending=False)[:10]

#Visualisasi
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_seller.index,x=top_10_seller.values)
plt.title('Top 10 Shop Revenue',fontsize=20)
plt.xlabel('Total Revenue',fontsize=17)
plt.ylabel('Shop Name',fontsize=17)
top_10_seller
#Toko apa yang paling laris?
top_10_seller = data.groupby(['shop_name'])['total_sales'].sum().sort_values(ascending=True)[:10]

#Visualisasi
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_seller.index,x=top_10_seller.values)
plt.title('Top 10 Least Shop Revenuw',fontsize=20)
plt.xlabel('Total Revenue',fontsize=17)
plt.ylabel('Shop Name',fontsize=17)
top_10_seller
#kategori apa yang paling laris?
top_10_most_category = data.groupby(['item_category_name'])['item_cnt_day'].sum().sort_values(ascending=False)[:10]

#Visualisasi 10 item paling laris
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_most_category.index,x=top_10_most_category.values)
plt.title('Top 10 Most Selling Category',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Category Name',fontsize=17)
#kategori apa yang paling tidak laris?
top_10_least_category = data.groupby(['item_category_name'])['item_cnt_day'].sum().sort_values(ascending=True)[:10]

#Visualisasi 10 item paling laris
plt.figure(figsize=(16,9))
sns.barplot(y=top_10_least_category.index,x=top_10_least_category.values)
plt.title('Top 10 Least Selling Category',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Category Name',fontsize=17)
#Merubah tipe data kolom date menjadi tipe date
data =  data.set_index('date')
data.head()
#Membuat data menjadi sebuah group dengan menghitung total sales setiap bulanya
train_arima = data.resample("M").sum() 
ts_sales = train_arima[["total_sales"]]
ts_sales.head()
#Menghilangkan data setelah oktober 2015 karena data setelah oktober tidak relevan, hanya sedikit data yang diambil
ts_sales = ts_sales[ts_sales.index <= pd.to_datetime('2015-10-31')]
#Viasualisasi data
plt.figure(figsize=(16,9))
plt.title('Total Item of the company')
plt.xlabel('Month')
plt.ylabel('Item')
plt.plot(ts_sales['total_sales'])
#Melihat apakah data kita stationary atau tidak dengan Dickey-Fuller Test
#Membuat fungsi
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
#Cek Stationary
test_stationarity(ts_sales['total_sales'])
#Visualisasi Trend dan Seasonlaity
item_cnt_dec = sm.tsa.seasonal_decompose(ts_sales['total_sales'],freq=12).plot()
#Visualisasi autocorrelation plot pada data item_cnt_day
sales_acf = sm.graphics.tsa.plot_acf(ts_sales['total_sales'], lags=12)
#Visualisasi autocorrelation plot pada data item_cnt_day
sales_acf = sm.graphics.tsa.plot_pacf(ts_sales['total_sales'], lags=12)
#Membuat nilai p,d,q dengan rentang 0 and 3
p = d = q = range(0, 2)

#Membuat iterasi nilai p,dq
pdq = list(itertools.product(p, d, q))


#Membuat seasonal dengan 12 bulan (karena yang paling terlihat / smooth)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#Melakukan pencarian nilai p,d,q dengan parameter AIC, semakin rendah semakin baik
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(ts_sales['total_sales'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=True,
                                            enforce_invertibility=True)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
#Kita masukkan kedalma model
mod = sm.tsa.statespace.SARIMAX(ts_sales['total_sales'],
                                order=(0,1,0),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 9))
plt.show()
#Melakukan prediksi dari bulan Desember 2014
pred = results.get_prediction(start=pd.to_datetime('2014-12-31'), dynamic=False)
pred_ci = pred.conf_int()
#Visualisasi prediksi
ax = ts_sales['2013-01-31':].plot(label = "observed", figsize=(16, 9))
pred.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Month')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

train_sarima_forecasted = pred.predicted_mean
train_sarima_truth = ts_sales['2014-12-31':]

#Menghiung RMSE
rmse_sarima = sqrt(mean_squared_error(train_sarima_truth, train_sarima_forecasted))
print("Root Mean Squared Error: ", rmse_sarima)
#Prediksi 3 total sales 3 bulan kedepan
pred_uc = results.get_forecast(steps=3)
pred_ci = pred_uc.conf_int()
ax = ts_sales['2013-01-31':].plot(label='observed', figsize=(16, 9))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Amount of Sales')
plt.legend()
plt.show()
#Reset index dulu untuk membuat date menjadi kolom
ts_sales = ts_sales.reset_index()
#Pertama kita harus merubah dates menjadi ds dan total_sales menjadi y
ts_sales.rename(columns={'date':'ds','total_sales':'y'},inplace=True)
#Membuat prediksi dengan Prophet
m_p = Prophet()
m_p.fit(ts_sales)
future = m_p.make_future_dataframe(periods = 3, freq = 'M')
prediction = m_p.predict(future)
prediction.tail(3)
#Visualisai Prediksi
m_p.plot(prediction)
plt.show()
#Visualisasi [Trends,Weekly]
m_p.plot_components(prediction)
plt.show()
#Melakukan Crossvalidation dengan training data 720 (2 tahun), prediksi 120 (4 bulan) dan interval waktunya 240 (8 bulan)
cv = cross_validation(m_p,initial='720 days', period='120 days', horizon = '240 days')
#Melihat hasil cv
cv.head()
#Melihat performance metrics
df_pm= performance_metrics(cv)
df_pm.head()
#Visualiasi RMSE
plot_cross_validation_metric(cv, metric='rmse')
plt.show()
pred_ci