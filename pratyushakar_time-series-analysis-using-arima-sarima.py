import warnings
warnings.filterwarnings("ignore")
#Data Manipulation and Treatment
import numpy as np
import pandas as pd
from datetime import datetime
#Plotting and Visualizations
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
from scipy import stats
import itertools
#Scikit-Learn for Modeling
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# statistics
#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# additional store data

store = pd.read_csv("../input/store.csv")
store.describe()
store.head()
# importing train data

train = pd.read_csv("../input/train.csv", parse_dates = True, low_memory = False, index_col = 'Date')
train.describe()
train.head()
# data extraction

train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear

train.info()
train.head()
# adding new variable

train['SalePerCustomer'] = train['Sales']/train['Customers']
train['SalePerCustomer'].describe()
train.isnull().sum()
train.fillna(0, inplace = True)
sns.set(style = "ticks")# to format into seaborn 
c = '#386B7F' # basic color for plots
plt.figure(figsize = (12, 6))

plt.subplot(311)
cdf = ECDF(train['Sales'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('Sales'); plt.ylabel('ECDF');

# plot second ECDF  
plt.subplot(312)
cdf = ECDF(train['Customers'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('Customers');

# plot second ECDF  
plt.subplot(313)
cdf = ECDF(train['SalePerCustomer'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('Sale per Customer');
# closed stores
train[(train.Open == 0) & (train.Sales == 0)]
# opened stores with zero sales
zero_sales = train[(train.Open != 0) & (train.Sales == 0)]
print("In total: ", zero_sales.shape)
zero_sales.head(5)
#print("Closed stores and days which didn't have any sales won't be counted into the forecasts.")

train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

print("In total: ", train.shape)
train=train.drop(columns=train[(train.Open == 1) & (train.Sales == 0)].index)
{"Mean":np.mean(train.Sales),"Median":np.median(train.Sales)}
train.Customers.describe()
{"Mean":np.mean(train.Customers),"Median":np.median(train.Customers)}
store.head()
# missing values?
store.isnull().sum()
# missing values in CompetitionDistance
store[pd.isnull(store.CompetitionDistance)]

store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
# no promo = no information about the promo?
_ = store[pd.isnull(store.Promo2SinceWeek)]
_[_.Promo2 != 0].shape
# replace NA's by 0
store.fillna(0, inplace = True)
print("Joining train set with an additional store information.")

# by specifying inner join we make sure that only those observations 
# that are present in both train and store sets are merged together
train_store = pd.merge(train, store, how = 'inner', on = 'Store')

print("In total: ", train_store.shape)
train_store.head(1000)

train_store.head(100)
train_store.groupby('StoreType')['Sales'].describe()
train_store.groupby('StoreType')['Customers', 'Sales'].sum()
# sales trends
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = c)
# sales trends
sns.factorplot(data = train_store, x = 'Month', y = "Customers", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', 
               color = c)
# sale per customer trends
sns.factorplot(data = train_store, x = 'Month', y = "SalePerCustomer", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = c)
# customers
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'DayOfWeek', # per store type in cols
               palette = 'plasma', 
               hue = 'StoreType',
               row = 'StoreType', # per store type in rows
               color = c)
# stores which are opened on Sundays
train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['Store'].unique()
# competition open time (in months)
train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
        (train_store.Month - train_store.CompetitionOpenSinceMonth)
    
# Promo open time
train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
        (train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0

# replace NA's by 0
train_store.fillna(0, inplace = True)

# average PromoOpen time and CompetitionOpen time per store type
train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()
# HeatMap
# Compute the correlation matrix 
# exclude 'Open' variable
corr_all = train_store.drop('Open', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask, annot = True, square = True, linewidths = 0.5, ax = ax, cmap = "BrBG", fmt='.2f')      
plt.show()
# sale per customer trends
sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Promo', 
               row = 'Promo2',
               hue = 'Promo2',
               palette = 'RdPu')
train_store.head()
# preparation: input should be float type
train['Sales'] = train['Sales'] * 1.0

# store types
sales_a = train[train.Store == 2]['Sales']
sales_b = train[train.Store == 85]['Sales']
sales_c = train[train.Store == 1]['Sales']
sales_d = train[train.Store == 13]['Sales']

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# store types
sales_a.resample('W').sum().plot(color = c, ax = ax1)
sales_b.resample('W').sum().plot(color = c, ax = ax2)
sales_c.resample('W').sum().plot(color = c, ax = ax3)
sales_d.resample('W').sum().plot(color = c, ax = ax4)
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# monthly
decomposition_a = seasonal_decompose(sales_a, model = 'additive', freq = 365)
decomposition_a.trend.plot(color = c, ax = ax1)

decomposition_b = seasonal_decompose(sales_b, model = 'additive', freq = 365)
decomposition_b.trend.plot(color = c, ax = ax2)

decomposition_c = seasonal_decompose(sales_c, model = 'additive', freq = 365)
decomposition_c.trend.plot(color = c, ax = ax3)

decomposition_d = seasonal_decompose(sales_d, model = 'additive', freq = 365)
decomposition_d.trend.plot(color = c, ax = ax4)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)
from scipy import stats
from scipy.stats import normaltest
def residual_plot(model):

    resid = model.resid
    print(normaltest(resid))
    # returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
    # the residual is not a normal distribution

    fig = plt.figure(figsize=(12,8))
    ax0 = fig.add_subplot(111)

    sns.distplot(resid ,fit = stats.norm, ax = ax0) # need to import scipy.stats

    # Get the fitted parameters used by the function
    (mu, sigma) = stats.norm.fit(resid)

    #Now plot the distribution using 
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Residual distribution')


    # ACF and PACF
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(model.resid, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(model.resid, lags=40, ax=ax2)
test_stationarity(sales_a)
first_diff_a = sales_a - sales_a.shift(1)
first_diff_a = first_diff_a.dropna(inplace = False)
test_stationarity(first_diff_a, window = 12)
test_stationarity(sales_b)
first_diff_b = sales_b - sales_b.shift(1)
first_diff_b = first_diff_b.dropna(inplace = False)
test_stationarity(first_diff_b, window = 12)
test_stationarity(sales_c)
first_diff_c = sales_c - sales_c.shift(1)
first_diff_c = first_diff_c.dropna(inplace = False)
test_stationarity(first_diff_c, window = 12)
test_stationarity(sales_d)
first_diff_d = sales_d - sales_d.shift(1)
first_diff_d = first_diff_d.dropna(inplace = False)
test_stationarity(first_diff_d, window = 12)
# figure for subplots
plt.figure(figsize = (12, 8))

# acf and pacf for A
plt.subplot(421); plot_acf(sales_a, lags = 50, ax = plt.gca(), color = c)
plt.subplot(422); plot_pacf(sales_a, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for B
plt.subplot(423); plot_acf(sales_b, lags = 50, ax = plt.gca(), color = c)
plt.subplot(424); plot_pacf(sales_b, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for C
plt.subplot(425); plot_acf(sales_c, lags = 50, ax = plt.gca(), color = c)
plt.subplot(426); plot_pacf(sales_c, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for D
plt.subplot(427); plot_acf(sales_d, lags = 50, ax = plt.gca(), color = c)
plt.subplot(428); plot_pacf(sales_d, lags = 50, ax = plt.gca(), color = c)

plt.show()
# figure for subplots
plt.figure(figsize = (12, 8))

# acf and pacf for A
plt.subplot(421); plot_acf(first_diff_a, lags = 50, ax = plt.gca(), color = c)
plt.subplot(422); plot_pacf(first_diff_a, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for B
plt.subplot(423); plot_acf(first_diff_b, lags = 50, ax = plt.gca(), color = c)
plt.subplot(424); plot_pacf(first_diff_b, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for C
plt.subplot(425); plot_acf(first_diff_c, lags = 50, ax = plt.gca(), color = c)
plt.subplot(426); plot_pacf(first_diff_c, lags = 50, ax = plt.gca(), color = c)

# acf and pacf for D
plt.subplot(427); plot_acf(first_diff_d, lags = 50, ax = plt.gca(), color = c)
plt.subplot(428); plot_pacf(first_diff_d, lags = 50, ax = plt.gca(), color = c)

plt.show()
arima_mod_a = sm.tsa.ARIMA(sales_a, (11,1,0)).fit(disp=False)
print(arima_mod_a.summary())
residual_plot(arima_mod_a)
sarima_mod_a = sm.tsa.statespace.SARIMAX(sales_a, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_a.summary())
residual_plot(sarima_mod_a)
print(sales_a.shape)
sales_a.head()
sales_a_reindex = sales_a.reindex(index=sales_a.index[::-1])
sales_a_reindex
mydata_a = sales_a_reindex
#mydata_a = sales_a_reindex.loc['2013-01-02':'2015-01-21']
#mydata_test = data.loc['2017-01-01':]
print(mydata_a)
temp_df =pd.DataFrame(mydata_a)
mydata_a = temp_df
sarima_mod_a_train = sm.tsa.statespace.SARIMAX(mydata_a, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_a_train.summary())
residual_plot(sarima_mod_a_train)
plt.figure(figsize=(50,10))
plt.plot(mydata_a, c='red')
plt.plot(sarima_mod_a_train.fittedvalues, c='blue')
plt.ylabel("Sales")
plt.xlabel("Time")
#forecast = sarima_mod_a_train.predict(start =mydata_a.loc['2015-01-21':], dynamic= True)  
#plt.plot(mydata_a.loc['2013-01-02':'2015-01-21'])
plt.figure(figsize=(30,10))
forecast = sarima_mod_a_train.predict(start = 625, end = 783, dynamic= False)  
plt.plot(mydata_a.iloc[1:625])
plt.plot(forecast, c = "red")
forecast
#pred_ci = forecast.conf_int()
#pred_ci.head()

#start_index = 624
#end_index = 784
#mydata_a['forecast'] = sarima_mod_a_train.predict(start = start_index, end= end_index, dynamic= True)  
#mydata_a[start_index:end_index][['sales', 'forecast']].plot(figsize=(12, 8))
arima_mod_b = sm.tsa.ARIMA(sales_b, (1,1,0)).fit(disp=False)
print(arima_mod_b.summary())
residual_plot(arima_mod_b)
sarima_mod_b = sm.tsa.statespace.SARIMAX(sales_b, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_b.summary())
residual_plot(sarima_mod_b)
print(sales_b.shape)
sales_b.head()
sales_b_reindex = sales_b.reindex(index=sales_b.index[::-1])
#sales_b_reindex.head(100)
mydata_b = sales_b_reindex
temp_df =pd.DataFrame(mydata_b)
mydata_b = temp_df
sarima_mod_b_train = sm.tsa.statespace.SARIMAX(mydata_b, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_b_train.summary())
residual_plot(sarima_mod_b_train)
plt.figure(figsize=(50,10))
plt.plot(mydata_b, c='red')
plt.plot(sarima_mod_b_train.fittedvalues, c='blue')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.figure(figsize=(30,10))
forecast = sarima_mod_b_train.predict(start = 755, end = 941, dynamic= False)  
plt.plot(mydata_b.iloc[1:755])
plt.plot(forecast, c = "red")
forecast
arima_mod_c = sm.tsa.ARIMA(sales_c, (11,1,0)).fit(disp=False)
print(arima_mod_c.summary())
residual_plot(arima_mod_c)
sarima_mod_c = sm.tsa.statespace.SARIMAX(sales_c, trend='n', order=(11,1,0)).fit()
print(sarima_mod_c.summary())
residual_plot(sarima_mod_c)
sales_c_reindex = sales_c.reindex(index=sales_c.index[::-1])
mydata_c = sales_c_reindex
temp_df =pd.DataFrame(mydata_c)
mydata_c = temp_df
sarima_mod_c_train = sm.tsa.statespace.SARIMAX(mydata_c, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_c_train.summary())
residual_plot(sarima_mod_c_train)
print(sales_c.shape)
sales_c.head()
plt.figure(figsize=(50,10))
plt.plot(mydata_c, c='red')
plt.plot(sarima_mod_c_train.fittedvalues, c='blue')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.figure(figsize=(30,10))
forecast = sarima_mod_c_train.predict(start = 625, end = 780, dynamic= False)  
plt.plot(mydata_c.iloc[1:625])
plt.plot(forecast, c = "red")
forecast
arima_mod_d = sm.tsa.ARIMA(sales_d, (11,1,0)).fit(disp=False)
print(arima_mod_d.summary())
residual_plot(arima_mod_d)
sarima_mod_d = sm.tsa.statespace.SARIMAX(sales_d, trend='n', order=(11,1,0),seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_d.summary())
residual_plot(sarima_mod_d)
print(sales_d.shape)
sales_d.head()
sales_d_reindex = sales_d.reindex(index=sales_d.index[::-1])
mydata_d = sales_d_reindex
temp_df =pd.DataFrame(mydata_d)
mydata_d = temp_df
sarima_mod_d_train = sm.tsa.statespace.SARIMAX(mydata_d, trend='n', order=(11,1,0),seasonal_order=(2,1,0,12)).fit()
print(sarima_mod_d_train.summary())
residual_plot(sarima_mod_d_train)
# Stote type D
plt.figure(figsize=(50,10))
plt.plot(mydata_d, c='red')
plt.plot(sarima_mod_d_train.fittedvalues, c='blue')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.figure(figsize=(30,10))
forecast = sarima_mod_d_train.predict(start = 495, end = 620, dynamic= False)  
plt.plot(mydata_d.iloc[1:495])
plt.plot(forecast, c = "red")
forecast