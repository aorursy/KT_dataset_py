import pandas as pd
import numpy as np
pd.set_option("display.max.columns",None)
pd.set_option("display.max.rows",None)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

%matplotlib inline
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew
import fbprophet
Prophet = fbprophet.Prophet

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import performance_metrics

import sys
sys.path.insert(0, '/kaggle/input/walmart-dataset')
import utility
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
path = '/kaggle/input/walmart-recruiting-store-sales-forecasting'
features = pd.read_csv(os.path.join(path, 'features.csv.zip'))
stores = pd.read_csv(os.path.join(path, 'stores.csv'))
test = pd.read_csv(os.path.join(path,'test.csv.zip'))
train = pd.read_csv(os.path.join(path,'train.csv.zip'))
train = train.drop(columns=['IsHoliday'],axis = 1)
features.head()


store_features = stores.merge(features,on = 'Store',how = 'inner')
store_data = train.merge(store_features,on = ['Store','Date'],how = 'right')
store_data.columns
dataset = store_data.drop(columns = ['Temperature','Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
       'MarkDown5','Unemployment'],axis = 1)
dataset.head()
dataset_for_prediction=dataset.dropna()
dataset_grouped = dataset_for_prediction.groupby(['Date','Store','Dept','CPI'])['Weekly_Sales'].sum().reset_index()
Store1 = dataset_grouped[dataset_grouped.Store.isin([1])]
Store1.head()
steps=0
dataset_for_prediction1= Store1.copy()
# dataset_for_prediction1['Actual_sales']=dataset_for_prediction1['Weekly_Sales'].shift(steps)
dataset_for_prediction1.head(3)
dataset_grouped = dataset_for_prediction1.groupby(['Date','Store','CPI'])['Weekly_Sales'].sum().reset_index()
dataset_for_prediction2=dataset_grouped.dropna()
dataset_for_prediction2['Date'] =pd.to_datetime(dataset_for_prediction2['Date'])
dataset_for_prediction2.index= dataset_for_prediction2['Date']
dataset_for_prediction2.Store.value_counts()
Store1 = dataset_for_prediction2.copy()
Store1 = Store1.rename(columns = {'Date':'ds','Weekly_Sales':'y'})
datetime_series = pd.to_datetime(Store1['ds'])

# create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)

Store1_data=Store1.set_index(datetime_index)
Store1_data.head(2)
SC =  Store1_data[['y']]
SC_with_regressors = utility.add_regressor(SC, Store1_data, varname='Store')
SC_with_regressors = utility.add_regressor(SC_with_regressors, Store1_data, varname='CPI')
# SC_with_regressors = utility.add_regressor(SC_with_regressors, Store1_data, varname='Dept')
data_train, data_test = utility.prepare_data(SC_with_regressors, 2012)
data_test.head()
m = Prophet(changepoint_prior_scale=0.05, interval_width=0.95,growth = 'linear',seasonality_mode = 'multiplicative', \
               yearly_seasonality=20, \
            weekly_seasonality=True, \
#             daily_seasonality=False,\
            changepoint_range=0.9)
m.add_seasonality('weekly', period=7, fourier_order=15)
m.add_regressor('Store')
# m.add_regressor('Dept')
m.add_regressor('CPI')
data_train.head()
data_train = data_train.rename(columns = {'index':'ds'})
data_test = data_test.rename(columns = {'index':'ds'})
data_train.ds = pd.to_datetime(data_train.ds)
data_test.ds = pd.to_datetime(data_test.ds)
data_test.head()

Store = SC_with_regressors[['Store']]
CPI = SC_with_regressors[['CPI']]
# Dept = SC_with_regressors[['Dept']]
m.fit(data_train)
future = m.make_future_dataframe(periods=len(data_test), freq='7D')
futures = utility.add_regressor_to_future(future, [Store,CPI])
forecast = m.predict(futures)
pd.plotting.register_matplotlib_converters()
f = m.plot_components(forecast)
plt.figure(figsize=(18, 6))
m.plot(forecast, xlabel = 'Date', ylabel = 'Weekly-Sales')
plt.title('Weekly Sales of Store1');

verif = utility.make_verif(forecast, data_train, data_test)
pd.plotting.register_matplotlib_converters()
def plot_verif1(verif, year=2012):
    """
    plots the forecasts and observed data, the `year` argument is used to visualise 
    the division between the training and test sets. 
    Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package
    year : integer
        The year used to separate the training and test set. Default 2017
    Returns
    -------
    f : matplotlib Figure object
    """
    
    f, ax = plt.subplots(figsize=(15,5))
    
    train = verif.loc[:str(year - 1),:]
    
    ax.plot(train.index, train.y, 'ko', markersize=3)
    
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    
    test = verif.loc[str(year):,:]
    
    ax.plot(test.index, test.y, 'ro', markersize=3)
    
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    
    ax.axvline(str(year), color='0.8', alpha=0.7)
    
    ax.grid(ls=':', lw=0.5)
    
    return f

# verif.loc[:,'yhat'] = verif.yhat.clip_lower(0)
# verif.loc[:,'yhat_lower'] = verif.yhat_lower.clip_lower(0)
f =  plot_verif1(verif,2012)
!pip install statsmodels
from statsmodels.tools.eval_measures import rmse
import datetime 
error=rmse(verif.loc[:'2012','y'].values,verif.loc[:'2012','yhat'].values)
error
# verif['yhat'].plot(legend=True, color='red', figsize=(10,8))
# dataset_for_prediction1['Weekly_Sales'].plot(legend=True, color='green', figsize=(20,8))
# dataset_for_prediction1['Weekly_Sales'].plot(legend=True, color='green', figsize=(20,8))
# verif['y'].plot(legend=True, color='blue', figsize=(10,8))

fig=verif['yhat'].plot(legend=True, color='red', figsize=(10,4))
fig=verif['y'].plot(legend=True, color='blue')
# fig.set_figheight(4)
plt.show()

Store1 = dataset_grouped[(dataset_grouped.Store==1)]
steps=-1
dataset_for_prediction1= Store1.copy()
dataset_for_prediction1['Actual_sales']=dataset_for_prediction1['Weekly_Sales'].shift(steps)
dataset_for_prediction1=dataset_for_prediction1.dropna()
dataset_for_prediction1['Date'] =pd.to_datetime(dataset_for_prediction1['Date'])
dataset_for_prediction1.index= dataset_for_prediction1['Date']
# dataset_for_prediction1.head()
X = dataset_for_prediction1[['Store','CPI','Weekly_Sales']]
y = dataset_for_prediction1[['Actual_sales']]
y.rename(columns={'Actual_sales':'Next_week_Sale'},inplace = True)
train_size=int(len(Store1) *0.7)
test_size = int(len(Store1)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()
import statsmodels.api as sm
seas_d=sm.tsa.seasonal_decompose(X['Weekly_Sales'],model='add',freq=7);
fig=seas_d.plot()
fig.set_figheight(4)
plt.show()
from statsmodels.tsa.stattools import adfuller
def test_adf(series, title=''):
    dfout={}
    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key,val in dftest[4].items():
        dfout[f'critical value ({key})']=val
    if dftest[1]<=0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)
y_test=y['Next_week_Sale'][:train_size].dropna()
test_adf(y_test, " Weekly Sales")
!pip install pmdarima
from pmdarima.arima import auto_arima
step_wise=auto_arima(train_y, 
 exogenous= train_X,
 start_p=1, start_q=1, 
 max_p=7, max_q=7, 
 d=1, max_d=7,
 trace=True, 
 error_action='ignore', 
 suppress_warnings=True, 
 stepwise=True)
step_wise.summary()
from statsmodels.tsa.statespace.sarimax import SARIMAX

model= SARIMAX(train_y, seasonal_order=(0,1,0,52),
 exog=train_X,
 order=(0,1,0),
 enforce_invertibility=False, enforce_stationarity=False)

results= model.fit()
results.summary()
%matplotlib inline
# steps = -1
predictions= results.predict(start =train_size, end=train_size+test_size+(steps)-1,exog=test_X)
act= pd.DataFrame(y.iloc[train_size:, :])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual_sales'] = act['Next_week_Sale']
predictions.rename(columns={0:'Pred_sales'}, inplace=True)
Final_predictions = pd.concat([predictions,act],axis=1)
# Final_predictions['Pred_sales'].plot(legend=True, color='red', figsize=(20,8))
# dataset_for_prediction2['Actual_sales'].plot(legend=True, color='blue', figsize=(20,8))
fig=Final_predictions['Pred_sales'].plot(legend=True, color='red', figsize=(10,4))
fig=dataset_for_prediction1['Actual_sales'].plot(legend=True, color='blue')
# fig.set_figheight(4)
plt.show()

from statsmodels.tools.eval_measures import rmse
error=rmse(predictions['Pred_sales'], predictions['Actual_sales'])
error
## New markdown cell
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
Store1 = Store1.groupby(['Date','Store','CPI'])['Weekly_Sales'].sum().reset_index()
Store1.head()
Store1.Date = pd.to_datetime(Store1.Date)
Store1.dtypes
# Store1.drop('Date',axis = 1, inplace=True)
Store1.index = Store1['Date']
Store1.head()
# Store1['sale_year'], Store1['sale_month'], Store1['sale_day'] = Store1.Date.dt.year,Store1.Date.dt.month,Store1.Date.dt.day
import re
#function from fasiai
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    if isinstance(fldnames,str): 
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)

add_datepart(Store1, 'Date')
Store1.head()
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_trn = int(len(Store1) *0.7)
x = Store1.drop('Weekly_Sales',axis = 1)
y = Store1[['Weekly_Sales']]
X_train, X_test = split_vals(x, n_trn)
y_train, y_test = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_test.shape
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 500, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt',0.5]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,3,5]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_
from statsmodels.tools.eval_measures import rmse

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
    rmse_err = rmse(predictions, test_labels['Weekly_Sales'])
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
    
    print('Model Performance')
    print(f'RMSE: {rmse_err}')
   
    return rmse

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
print(random_accuracy)
import math
from statsmodels.tools.eval_measures import rmse

np.random.seed(500)
# m = RandomForestRegressor(n_estimators=39, n_jobs=-1, min_samples_leaf=3,oob_score= True,max_features=0.5) #oob_score=True
m = RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_split=5,min_samples_leaf= 1,max_features=0.5,max_depth=100,bootstrap=False) #oob_score=True
m.fit(X_train,y_train)
predictions = m.predict(X_test)
error=rmse(predictions, y_test['Weekly_Sales'])
error
# print_score(m)
predictions_df = pd.DataFrame(predictions)
predictions_df.rename(columns = {0:'Pred'},inplace = True)
predictions_df.head(10)
y_test['that'] = list(predictions_df['Pred'])

final_rf = pd.concat([y_test,X_test],axis = 1)
# final_rf['Weekly_Sales'] = np.exp(final_rf['Weekly_Sales_log'])
final_rf.head()
pd.plotting.register_matplotlib_converters()
fig=y_train['Weekly_Sales'].plot(legend=True, color='blue', figsize=(15,4))
fig=final_rf['that'].plot(legend=True, color='red')
fig=final_rf['Weekly_Sales'].plot(legend=True, color='blue')

from statsmodels.tools.eval_measures import rmse
error=rmse(final_rf['that'], final_rf['Weekly_Sales'])
error