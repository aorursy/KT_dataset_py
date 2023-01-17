# Data Processing
import numpy as np
import pandas as pd
from pandas import datetime

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
import matplotlib.gridspec as gridspec
from IPython.display import display
%matplotlib inline

# Data Modeling
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA,ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima # for determining ARIMA orders
from fbprophet import Prophet

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm

# Data Evaluation
from sklearn.metrics import mean_squared_error

# Statistics
from statsmodels.distributions.empirical_distribution import ECDF

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Warning ignore
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/rossmann-store-sales/train.csv")
store = pd.read_csv("../input/rossmann-store-sales/store.csv")
test = pd.read_csv("../input/rossmann-store-sales/test.csv", parse_dates = True, index_col = 'Date')
train.head()
# Change datatype of InvoiceDate as datetime type
train['Date'] = pd.to_datetime(train['Date'])
# data extraction
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day
train['WeekOfYear'] = train['Date'].dt.weekofyear

test['Year'] = test.index.year
test['Month'] = test.index.month
test['Day'] = test.index.day
test['WeekOfYear'] = test.index.weekofyear
# adding new variable
train['SalePerCustomer'] = train['Sales']/train['Customers']
train['SalePerCustomer'].describe()
plt.figure(figsize = (12, 6))

plt.subplot(311)
cdf = ECDF(train['Sales'])
plt.plot(cdf.x, cdf.y, label = "statmodels");
plt.title('Sales'); plt.ylabel('ECDF');

# plot second ECDF  
plt.subplot(312)
cdf = ECDF(train['Customers'])
plt.plot(cdf.x, cdf.y, label = "statmodels");
plt.title('Customers');

# plot second ECDF  
plt.subplot(313)
cdf = ECDF(train['SalePerCustomer'])
plt.plot(cdf.x, cdf.y, label = "statmodels");
plt.title('Sale per Customer');
plt.subplots_adjust(hspace = 0.8)
sns.distplot(train['Sales'])
# Closed Stores with zero sales
train[(train.Open == 0)]
# Opened stores with zero sales
len(train[(train.Open == 1) & (train.Sales == 0)])
# Closed stores and days which didn't have any sales won't be counted into the forecasts.
train = train[(train["Open"] != 0) & (train['Sales'] != 0)]
train.head()
# missing values?
store.isnull().sum()
# missing values in CompetitionDistance
store[pd.isnull(store.CompetitionDistance)]
# fill NaN with a median value (skewed distribuion)
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
# replace NA's by 0
store.fillna(0, inplace = True)
# by specifying inner join we make sure that only those observations 
# that are present in both train and store sets are merged together
train_store = pd.merge(train, store, how = 'inner', on = 'Store')

test_store = pd.merge(test, store, how = 'inner', on = 'Store')

train_store.head()
train_store.groupby('StoreType')['Sales'].describe()
train_store.groupby('StoreType')['Customers', 'Sales'].sum()
# sales trends
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo') 
# sales trends
sns.factorplot(data = train_store, x = 'Month', y = "Customers", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo') 
# sale per customer trends
sns.factorplot(data = train_store, x = 'Month', y = "SalePerCustomer", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo')
# customers
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'DayOfWeek', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'StoreType')
# competition open time (in months)
train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
        (train_store.Month - train_store.CompetitionOpenSinceMonth)  
# Promo open time
train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
        (train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0

# test_store['CompetitionOpen'] = 12 * (test_store['Year'] - test_store['CompetitionOpenSinceYear']) + (test_store['Month'] - test_store['CompetitionOpenSinceMonth'])
# test_store['PromoOpen'] = 12 * (test_store['Year'] - test_store['Promo2SinceYear']) + (test_store['WeekOfYear'] - test_store['Promo2SinceWeek']) / 4.0


# replace NA's by 0
train_store.fillna(0, inplace = True)

# average PromoOpen time and CompetitionOpen time per store type
train_store[['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()
# Compute the correlation matrix 
# exclude 'Open' variable
corr_all = train_store.drop('Open', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
plt.show()
# sale per customer trends
sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Promo', 
               row = 'Promo2',
               hue = 'Promo',   # SPECIAL
               palette = 'RdPu') 
# to numerical
mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

train['StateHoliday'] = train['StateHoliday'].replace(mappings).astype('int64')

test['StateHoliday'] = test['StateHoliday'].replace(mappings).astype('int64')

store['StoreType'] = store['StoreType'].replace(mappings).astype('int64')
store['Assortment'] = store['Assortment'].replace(mappings).astype('int64')
store.drop('PromoInterval', axis = 1, inplace = True)

train_store = pd.merge(train, store, how = 'inner', on = 'Store')
test_store = pd.merge(test, store, how = 'inner', on = 'Store')
# Choose 1 store with type a, namely store 2
sales_2 = train[train['Store'] == 2][['Sales', 'Date','StateHoliday','SchoolHoliday']]
sales_2['Date'].sort_index(ascending = False, inplace=True)
a = sales_2.set_index('Date').resample('W').sum()
a
plt.figure(figsize=(12, 5))
sns.lineplot(x=a.index, y=a['Sales'])
f, (ax1, ax2) = plt.subplots(2, figsize = (12, 6))

# monthly
decomposition_a = seasonal_decompose(sales_2['Sales'], model = 'additive', freq = 365)
decomposition_a.observed.plot(ax = ax1)
ax1.set_ylabel('OBSERVED')
decomposition_a.trend.plot(ax = ax2)
ax2.set_ylabel('TREND')
f.subplots_adjust(hspace = 0.5)
fig = plt.figure(constrained_layout=True, figsize=(15, 4))
grid = gridspec.GridSpec(nrows=1, ncols=2,  figure=fig)

# acf and pacf for A
ax1 = fig.add_subplot(grid[0, 0])
plot_acf(sales_2['Sales'], lags = 50, ax=ax1);

# acf and pacf for A
ax1 = fig.add_subplot(grid[0, 1])
plot_pacf(sales_2['Sales'], lags = 50, ax=ax1);

plt.show();
# adfuller helps us to determine the right model for analysis. 
# For example, the returned value from adf_test show 'Fail to reject the null hypothesis', it means we should make differencing.

from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
adf_test(sales_2['Sales'])
x = sales_2.set_index('Date').loc[:'2015-06-30']
y = sales_2.set_index('Date').loc['2015-07-01':]
model = AR(x['Sales'])
AR1fit = model.fit(method='mle')
#print(f'Lag: {AR1fit.k_ar}')
#print(f'Coefficients:\n{AR1fit.params}')

# This is the general format for obtaining predictions
start=len(x)
end=len(x)+len(y)-1
predictions1 = AR1fit.predict(start=start, end=end, dynamic=False)

predictions1.index = y.index

y['Sales'].plot()
predictions1.plot(label='prediction');
plt.legend()
plt.show()

print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y['Sales'], predictions1)))
# auto_arima help us choose the optimal model, sometime manually tweaking model hyperparameters yeild better result.
auto_arima(x['Sales'],seasonal=False).summary()
model = ARMA(x['Sales'], order=(5,5))
results = model.fit()

start=len(x)
end=len(x)+len(y)-1

predictions1 = results.predict(start=start, end=end, dynamic=False)
predictions1.index = y.index

y['Sales'].plot()
predictions1.plot(label='prediction');

print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y['Sales'], predictions1)))
# Adding exorgenous variable may help accuracy improvement. Let's see
# It doesnt improve as the store usually closed on StateHoliday or SchoolHoliday and sales may not escalated even if it opens.
auto_arima(x['Sales'], exorgenous=x[['StateHoliday','SchoolHoliday']],seasonal=False).summary()
model = ARMA(x['Sales'], order=(5,5))
results = model.fit()

# This is the general format for obtaining predictions
start=len(x)
end=len(x)+len(y)-1
exog_forecast = x[['StateHoliday','SchoolHoliday']]
predictions1 = results.predict(start=start, end=end, exog=exog_forecast, dynamic=False)

predictions1.index = y.index

y['Sales'].plot()
predictions1.plot(label='prediction');
plt.legend()

print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y['Sales'], predictions1)))
# As we talk above, we may be interested in the fact that event or seasonality can influence sale of store.
# However, in this case, adding seasonality worsen model. Thus, there is no clear seasonal component in this case.

# https://alkaline-ml.com/pmdarima/tips_and_tricks.html#setting-m
# m = 7(daily), 12(monthly), 52(weekly)
auto_arima(x['Sales'],seasonal=True, m=7).summary()
model = SARIMAX(x['Sales'],order=(5, 0, 3),seasonal_order=(0, 0, 1, 7))
results = model.fit()

start=len(x)
end=len(x)+len(y)-1
predictions1 = results.predict(start=start, end=end, dynamic=False)

predictions1.index = y.index

y['Sales'].plot()
predictions1.plot(label='prediction');
plt.legend()

print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y['Sales'], predictions1)))
state_dates = x[(x.StateHoliday == 0) | (x.StateHoliday == 1) & (x.StateHoliday == 2)].reset_index()['Date'].values
school_dates = x[(x['SchoolHoliday'] == 1)].reset_index()['Date'].values

state = pd.DataFrame()
state['ds'] = pd.to_datetime(state_dates)
state['holiday'] = 'state_holiday'

school = pd.DataFrame()
school['ds'] = pd.to_datetime(school_dates)
school['holiday'] = 'school_holiday'

holidays = pd.concat((state, school))
TRAIN = x.reset_index()[['Date', 'Sales']]
TRAIN.columns = ['ds', 'y']
# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width = 0.95, 
                   holidays = holidays)
my_model.fit(TRAIN)
# dataframe that extends into future 6 weeks 
future_dates = my_model.make_future_dataframe(periods=31)
print("Last day to forecast.")
future_dates.tail(1)
# predictions
forecast = my_model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
a = forecast[['ds','yhat']]
b = y['Sales'].reset_index()
table = a.set_index('ds').join(b.set_index('Date')).dropna()
table.columns = ['yhat', 'y']
display(table.head(1))

table['yhat'].plot()
table['y'].plot()
plt.legend();
plt.show()

print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(table['yhat'], table['y'])))
# visualizing predicions
my_model.plot(forecast);
my_model.plot_components(forecast);
train_store['CompetitionOpen'] = 12 * (train_store['Year'] - train_store['CompetitionOpenSinceYear']) + (train_store['Month'] - train_store['CompetitionOpenSinceMonth'])
train_store['PromoOpen'] = 12 * (train_store['Year'] - train_store['Promo2SinceYear']) + (train_store['WeekOfYear'] - train_store['Promo2SinceWeek']) / 4.0

test_store['CompetitionOpen'] = 12 * (test_store['Year'] - test_store['CompetitionOpenSinceYear']) + (test_store['Month'] - test_store['CompetitionOpenSinceMonth'])
test_store['PromoOpen'] = 12 * (test_store['Year'] - test_store['Promo2SinceYear']) + (test_store['WeekOfYear'] - test_store['Promo2SinceWeek']) / 4.0
# Sorting dataframe according to datatime, the oldest is on top, the most recent is at the bottom.
train_store['Date'].sort_index(ascending = False, inplace=True)
def rmsle(y_pred, y):
    return np.sqrt(mean_squared_error(y_pred, y))

def model_check (estimators):
    model_table = pd.DataFrame()
    row_index = 0
    
    for est in estimators:
        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name
        
        est.fit(x_train, y_train)
        y_pred = est.predict(x_test)
        model_table.loc[row_index, 'Test Error'] = rmsle(y_pred, y_test)
        
        row_index += 1
        
        model_table.sort_values(by=['Test Error'],
                            ascending=True,
                            inplace=True)
    return model_table
# MODELS
lr = LinearRegression()
ls = Lasso()
GBoost = GradientBoostingRegressor(random_state = 0)
XGBoost = XGBRegressor(random_state = 0, n_job=-1)
LGBM = LGBMRegressor(random_state = 0, n_job=-1)
# Training dataset is separated into train_a and test_a.
# traing_a train data from 2013 till 2015-06-30, while test_a contain  data from 2015-07-01 till 2015-07-31.

train_a = train_store.set_index('Date').loc[:'2015-06-30']
test_a = train_store.set_index('Date').loc['2015-07-01':]

x_train = train_a.drop(['Sales', 'Customers'], axis=1)
y_train = train_a['Sales']
x_test = test_a.drop(['Sales', 'Customers'], axis=1)
y_test = test_a['Sales']
estimators = [lr, ls, GBoost, XGBoost, LGBM]
model_check(estimators)

# This part is different from the above. This particularly examinze the prediction power on store 2 only.

train_a = train_store[train_store['Store']==2].set_index('Date').loc[:'2015-06-30']
test_a = train_store[train_store['Store']==2].set_index('Date').loc['2015-07-01':]

x_train = train_a.drop(['Sales', 'Customers','SalePerCustomer'], axis=1)
y_train = train_a['Sales']
x_test = test_a.drop(['Sales', 'Customers', 'SalePerCustomer'], axis=1)
y_test = test_a['Sales']
XGBoost = XGBRegressor(random_state = 0, n_job=-1).fit(x_train, y_train)
y_pred = XGBoost.predict(x_test)
LGBM = LGBMRegressor(random_state = 0, n_job=-1).fit(x_train, y_train)
y_pred = LGBM.predict(x_test)
xgb.plot_importance(XGBoost)
lightgbm.plot_importance(LGBM)
