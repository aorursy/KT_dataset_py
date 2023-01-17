import os
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import joblib
import datetime as dt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima_model import ARIMA
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
sales_train_eval = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales_train = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sample_sub = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
sales_train.head()
#Seperate Date Variables and others 
date_vars = sales_train.columns[6:] 
other_vars = sales_train.columns[:6]
print(date_vars,other_vars)
#Downcast in order to save memory
def downcast(df):
    cols = df.dtypes.index.tolist() #numbers of col: ['d_1814', 'd_1815',
    types = df.dtypes.values.tolist() #datatype in numbers of col: [dtype('int64'), dtype('int64'), 
    #-> for every column,their datatypes are all int64
    for i in range(len(types)): 
        if 'int' in str(types[i]):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
                
        elif 'float' in str(types[i]): 
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
                
        elif types[i] == np.object: #can be dates or categories 
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df  
            

sales_train = downcast(sales_train)
sell_prices = downcast(sell_prices)
calendar = downcast(calendar)
# sales_df = sales_train.melt(id_vars = other_vars, value_vars = date_vars, var_name = "Date")
#Join calendar data 

# calendar_to_join = calendar[['date', 'd','wm_yr_wk']]
# sales_df = pd.merge(sales_df, calendar_to_join, left_on = 'Date'  , right_on = 'd', suffixes=('_sales', '_cal')).drop(['Date'], axis = 1)
# sales_df = pd.merge(sales_df, sell_prices, on=['store_id','item_id','wm_yr_wk'], how='left') 
# sales_df.head()
sales_train.shape
# Only preserve sales for days(d_1, d_2, ...)
df_day = sales_train.copy()
df_day.drop(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace = True)
df_day.describe()
df_day.loc['Ttl_Daily_Sales'] = df_day.sum() #add up the sum of the total daily sales for all items

plt.figure(figsize=(12,8))
sns.distplot(df_day.loc["Ttl_Daily_Sales"], bins=50, kde=False)
plt.title("Ttl Daily Sales Histogram", fontsize=14)
plt.show()
#find the 10th lowest sales 
df_day_trans = df_day.transpose()
smallest = df_day_trans.nsmallest(10, 'Ttl_Daily_Sales')
smallest['Ttl_Daily_Sales']
#Row number of each state
state_pie = sales_train['state_id'].value_counts().rename_axis('state_name').reset_index(name='counts')

#percentage of row numbers of each state
print(sales_train['state_id'].value_counts(normalize=True)) 

plt.pie(state_pie['counts'], labels= state_pie['state_name'], autopct = '%1.1f%%')


state = sales_train.groupby('state_id', axis = 0).sum()
state
state_trans = state.transpose()
state_trans.head()
state_trans['date'] = pd.date_range(start='1/29/2011', periods= len(state_trans), freq='D')
state_trans.set_index('date', drop = True, inplace = True)
state_trans.sort_index(inplace=True)
state_month = state_trans.groupby(pd.Grouper(freq='1M')).sum()
state_month.head(20)
state_month.plot(title = "Monthly Sales accross States")
#Method 2

state_trans['date'] = pd.date_range(start='1/29/2011', periods= len(state_trans), freq='D')
state_trans.set_index('date', drop = True, inplace = True)
state_trans.sort_index(inplace=True)
state_trans.index.to_period("M")
state_month.plot(title = "Monthly Sales accross States")
#Sales of each store
store_pie = sales_train.groupby('store_id').sum().T
store_pie.loc['ttl_sales'] = store_pie.sum()

plt.pie(store_pie.loc['ttl_sales'],labels = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2',
       'WI_3'], autopct = '%1.1f%%')
store = sales_train.groupby('store_id', axis = 0).sum().reset_index().set_index('store_id').T
store['date'] = pd.date_range(start='1/29/2011', periods= len(store), freq='D')
store.set_index('date', drop = True, inplace = True)
store.sort_index(inplace=True)
store_month = store.groupby(pd.Grouper(freq = '1M')).sum()
store_month.head(20)
store_month.plot(title = "Monthly Sales accross Store")
series = state_month['CA']
result = seasonal_decompose(series, model='additive')
fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(10,8))
print(result.trend.plot(ax=axes[0]))
print(result.seasonal.plot(ax=axes[1]))
print(result.resid.plot(ax=axes[2]))
print(result.observed.plot(ax=axes[3]))
axes[0].set_ylabel('trend')
axes[1].set_ylabel('seasonal')
axes[2].set_ylabel('resid')
axes[3].set_ylabel('observed')

plt.legend()
train_dataset = sales_train[date_vars[-100: -30]]
val_dataset = sales_train_eval[date_vars[-30:]]
sales_train_eval.columns
print(train_dataset.columns)
print(val_dataset.columns)
fig = make_subplots(rows = 2, cols = 1)


#first product sales from 1814 - 1883
fig.add_trace(
    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[0].values,  marker=dict(color="dodgerblue"),showlegend=False, 
               name="Original signal"),
    row=1, col=1
)

#first product sales from 1884 - 1913
fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', marker=dict(color="darkorange"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)



#second product sales from 1814 - 1883
fig.add_trace(
    go.Scatter(x=np.arange(70), mode='lines', y=train_dataset.loc[1].values,  marker=dict(color="dodgerblue"),showlegend=False, 
               name="Original signal"),
    row=2, col=1
)

#second product sales from 1884 - 1913
fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', marker=dict(color="darkorange"), showlegend=False,
               name="Denoised signal"),
    row=2, col=1
)

def moving_average(days, train_dataset):
    predictions = []
    for i in range(days):
        if i == 0:
            predictions.append(np.mean(train_dataset[train_dataset.columns[-28:]].values, axis=1))
            #when i is 0, average the 30 days sales for every product respectively

        elif i < 28 and i > 0:
            predictions.append(
                            (np.sum(train_dataset[train_dataset.columns[-28+i:]].values,axis =1) 
                          + np.sum(predictions[:i], axis=0)) /28  )
                                
            #if i is i, we calculate the latest 29 days of average sales, and average it with predictions 
        elif i >= 28:
            predictions.append(np.mean([predictions[i-28:i]], axis=0))
        
    predictions_array = np.transpose(np.array([row.tolist() for row in predictions]))
    
    return predictions_array

# y_1 = val_dataset[val_dataset.columns[-30:]]
# RMSE = mean_squared_error(y_1, predictions_1)
# RMSE
validation = sales_train[date_vars[-28:]]
pred_val = moving_average(28, validation)
sample_sub.iloc[30490:,1:] = pred_val
sample_sub
evaluation = sales_train_eval[sales_train_eval.columns[-28:]]
pred_eval = moving_average(28, evaluation)
sample_sub.iloc[:30490,1:] = pred_eval
sample_sub
filename = 'M5_2.csv'

sample_sub.to_csv(filename,index=False)

print('Saved file: ' + filename)
sample_sub
from IPython.display import FileLink
FileLink(r'M5_2.csv')
def Explo_smoothing(train_data):
    exp_predictions = []
    for rows in range(len(train_data)): 
        fit1 = ExponentialSmoothing(train_data.iloc[rows].values, seasonal_periods=28).fit(smoothing_level = 0.2)
        exp_predictions.append(fit1.forecast(28))
    return exp_predictions
pred_exp = Explo_smoothing(validation)
pred_evl_exp = Explo_smoothing(evaluation)
sample_sub.iloc[:30490,1:] = pred_evl_exp
sample_sub.iloc[30490:,1:] = pred_exp
sample_sub
filename = 'M5_3.csv'

sample_sub.to_csv(filename,index=False)

print('Saved file: ' + filename)
from IPython.display import FileLink
FileLink(r'M5_3.csv')
