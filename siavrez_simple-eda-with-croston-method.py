from pandas.plotting import scatter_matrix, andrews_curves, autocorrelation_plot, lag_plot

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.api import acf, pacf, graphics

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.robust import mad

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import plotly.tools as tls

import cufflinks as cf

import plotly

from IPython.display import HTML

import matplotlib.pyplot as plt

from scipy.signal import butter

from itertools import cycle

from scipy import signal

import seaborn as sns

import pandas as pd 

import numpy as np

import warnings

import os

import gc



pd.set_option('max_columns', 100)

pd.set_option('max_rows', 50)



cf.go_offline()

py.init_notebook_mode()

cf.getThemes()



cf.set_config_file(theme='space')

warnings.simplefilter('ignore')

pd.plotting.register_matplotlib_converters()

sns.mpl.rc('figure',figsize=(16, 6))

sns.set_style('whitegrid')

sns.set_context('poster')

palette = sns.color_palette('mako_r', 6)

%matplotlib inline

plt.rcParams['figure.figsize'] = (16,6)

plt.rcParams['axes.titlesize'] = 16
base = os.path.abspath('/kaggle/input/m5-forecasting-accuracy/')

sell_prices = pd.read_csv(os.path.join(base + '/sell_prices.csv'))

calendar  = pd.read_csv(os.path.join(base + '/calendar.csv'))

sales_train_validation  = pd.read_csv(os.path.join(base + '/sales_train_validation.csv'))

submission_file = pd.read_csv(os.path.join(base + '/sample_submission.csv'))

print(f'Shape of Data Files: \n\n calendar        = {calendar.shape} \n sales_train     = {sales_train_validation.shape} \n sell_prices     = {sell_prices.shape} \n submission_file = {submission_file.shape}')
sell_prices.head()
sell_prices.info()
sell_prices.describe()
print('Price Change for six random Item')

items_sample = sell_prices.item_id.sample(6, random_state=2020)

for df, s in sell_prices[sell_prices.item_id.isin(items_sample)].groupby(['item_id']):

    s.reset_index()['sell_price'].pct_change().iplot(theme='space', margin=(10, 10, 10, 30) ,dimensions=(800,150), xTitle='percent', yTitle='Days', title = f'price change of an item', mode='lines+markers', size=0.5)

print('Descriptive Statistics for Item prices')

sell_prices.groupby(['item_id'])['sell_price'].agg(['min', 'max', 'mean', 'count']).head(20)
print('Descriptive Statistics for Item prices by Store')

sell_prices.groupby(['store_id','item_id'])['sell_price'].agg(['min', 'max', 'mean', 'count']).head(20)
calendar.head()
calendar.info()
calendar.describe()
sales_train_validation.head()
sales_train_validation.info()
sales_train_validation.describe()
print(f'Number of unique States      : {sales_train_validation.state_id.nunique()}')

print(f'Number of unique Stores      : {sales_train_validation.store_id.nunique()}')

print(f'Number of unique Categories  : {sales_train_validation.cat_id.nunique()  }')

print(f'Number of unique Items       : {sales_train_validation.item_id.nunique()}')

print(f'Number of unique Sale Prices : {sales_train_validation.id.nunique()}')
print(f'Unique States      : {sales_train_validation.state_id.unique()}')

print(f'Unique Stores      : {sales_train_validation.store_id.unique()}')

print(f'Unique Categories  : {sales_train_validation.cat_id.unique()  }')
print(f'Number of unique Items per Store :')

sales_train_validation.groupby('store_id')['item_id'].agg(['count'])

days = [c for c in sales_train_validation.columns if 'd_' in c]

print(f'Number of Days in Validation Data : {len(days)}')
print(f'Minimum of Max sales in ONE Day : {sales_train_validation.loc[:,days].T.max().min()} \nMaximum of Max sales in ONE Day : {sales_train_validation.loc[:,days].T.max().max()}')
print(f'Maximum of Min sales in ONE Day : {sales_train_validation.loc[:,days].T.min().max()} \nMinimum of Min sales in ONE Day : {sales_train_validation.loc[:,days].T.min().min()}')
x, y = np.unique(sales_train_validation.loc[:,days].values.ravel(), return_counts=True)

counts = pd.DataFrame(y, index=x, columns=['Items Sold']).reset_index()

counts.columns = ['Items Sold', 'Day Count']

print('Number of Days with Count of each Items sold aggregated by Items')

counts.T
sales_sum_by_store = sales_train_validation.groupby(['store_id']).sum().T.reset_index(drop = True)

sales_mean_by_store = sales_train_validation.groupby(['store_id']).mean().T.reset_index(drop = True) 



print('Sales aggregated by different Stores')

sales_sum_by_store.iplot(kind='box',  margin=(10, 10, 10, 40) ,dimensions=(900,500), title = 'Total Sales by Store ID', xTitle = 'Store ID', yTitle = 'Total Sales')

sales_mean_by_store.iplot(kind='box',  margin=(10, 10, 10, 40) ,dimensions=(900,500), title = 'Average Sales by Store ID', xTitle = 'Store ID', yTitle = 'Average Sales')

sales_sum_by_state = sales_train_validation.groupby(['state_id']).sum().T.reset_index(drop = True)

sales_mean_by_state = sales_train_validation.groupby(['state_id']).mean().T.reset_index(drop = True) 



print('Sales aggregated by different States')

sales_sum_by_state.iplot(kind='box', margin=(10, 10, 10, 40) ,dimensions=(900,500), title = 'Total Sales by State', xTitle = 'State', yTitle = 'Total Sales')

sales_mean_by_state.iplot(kind='box',  margin=(10, 10, 10, 40) ,dimensions=(900,500), title = 'Average Sales by State', xTitle = 'State', yTitle = 'Average Sales')

sales_sum_by_cat = sales_train_validation.groupby(['cat_id']).sum().T.reset_index(drop = True)

sales_mean_by_cat = sales_train_validation.groupby(['cat_id']).mean().T.reset_index(drop = True) 



print('Sales aggregated by different Categories')

sales_sum_by_cat.iplot(kind='box', margin=(10, 10, 10, 40) ,dimensions=(900,500), title = 'Total Sales by Category', xTitle = 'Category', yTitle = 'Total Sales')

sales_mean_by_cat.iplot(kind='box',  margin=(10, 10, 10, 40) ,dimensions=(900,500), title = 'Average Sales by Category', xTitle = 'Category', yTitle = 'Average Sales')

department_mean_sale = pd.DataFrame(sales_train_validation.groupby(['dept_id']).mean().mean()).reset_index(drop=True).reset_index()

department_mean_sale.columns = ['Days', 'Sale']

fig = px.scatter(department_mean_sale, x = 'Days', y = 'Sale',color='Sale', trendline='lowess', template='plotly_dark',

                title='Average Total Sales of Item Per Day Over Time')

fig.update_layout(

    margin=dict(l=10, r=10, t=40, b=20),

)

fig.show()
days_dates = calendar.set_index('d')

days_item = sales_train_validation.loc[sales_train_validation['id'] == 'FOODS_3_825_WI_3_validation'].set_index('id')[days].T

item = days_item.merge(days_dates, left_index=True, right_index=True).set_index('date')

item.rename(columns = ({'FOODS_3_825_WI_3_validation' : 'item_'}), inplace=True)

item.iloc[:, 0].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,300), title = 'Sales of an item', xTitle='Year')
samples = sales_train_validation.loc[:, days].sample(6, random_state=2020).T

samples = days_dates.merge(samples, left_index=True, right_index=True).set_index('date')

samples.iloc[:, -6:].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,500), title = 'Sales of different items', xTitle='Year')
samples = sales_train_validation.loc[:, days].sample(6, random_state=2020).T.rolling(28).mean().fillna(0)

samples = days_dates.merge(samples, left_index=True, right_index=True).set_index('date')

samples.iloc[:, -6:].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,500), title = 'Rolling mean of Sales Window : 28', xTitle='Year')
samples = sales_train_validation.loc[:, days].sample(6, random_state=2020).T.rolling(28).std().fillna(0)

samples = days_dates.merge(samples, left_index=True, right_index=True).set_index('date')

samples.iloc[:, -6:].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,500), title = 'Rolling std of Sales Window : 28', xTitle='Year')
samples = sales_train_validation.loc[:, days].sample(6, random_state=2020).T.rolling(28).median().fillna(0)

samples = days_dates.merge(samples, left_index=True, right_index=True).set_index('date')

samples.iloc[:, -6:].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,500), title = 'Rolling median of Sales Window : 28', xTitle='Year')
items = sales_train_validation['id'].sample(6, random_state=2020)

days_item = sales_train_validation.loc[sales_train_validation['id'].isin(items)].set_index('id')[days].T

items = days_item.merge(days_dates, left_index=True, right_index=True).set_index('date')

items.rename(columns = ({'HOUSEHOLD_2_026_CA_2_validation' : 'item_0',

                        'FOODS_3_135_TX_1_validation' :      'item_1',

                        'HOUSEHOLD_1_301_WI_1_validation' :  'item_2',

                        'HOBBIES_2_001_WI_3_validation' :    'item_3',

                        'HOUSEHOLD_1_371_WI_3_validation' :  'item_4',

                        'HOUSEHOLD_2_238_WI_3_validation' :  'item_5',

                       }), inplace=True)

items.groupby('wday').mean().iloc[:, :6].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,400), title = 'Average Sales by week day', xTitle='WeekDays', mode='lines+markers')
items.groupby('month').mean().iloc[:, :6].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,500), title = 'Average Sales by month', xTitle='Month', mode='lines+markers')
items.groupby('year').mean().iloc[:, :6].iplot(theme='space', margin=(10, 10, 10, 40) ,dimensions=(800,500), title = 'Average Sales by year', xTitle='Year', mode='lines+markers')
sales_train_validation['sum'] = sales_train_validation.sum(axis=1, numeric_only=True)
sales_train_validation.groupby('dept_id')['sum'].mean().iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,300), title = 'Average Sales by Department', xTitle='Department', mode='lines+markers')
sales_train_validation.groupby('store_id')['sum'].mean().iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,300), title = 'Average Sales by Store', xTitle='Store', mode='lines+markers')
sales_train_validation.groupby('item_id')['sum'].mean().sample(frac=0.01, random_state=2020).iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,300), title='Average Sales By Item 0.01 of Data', xTitle='Item', mode='lines+markers')
sales_train_validation.groupby('cat_id')['sum'].mean().iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,300), title='Average Sales By Category', xTitle='Category', mode='lines+markers')
sales_train_validation.groupby('state_id')['sum'].mean().iplot(theme='space', color='gray',  kind='bar', margin=(10, 10, 10, 40) ,dimensions=(500,300), xTitle='Average By State', mode='lines+markers')
day_sums = pd.DataFrame(sales_train_validation.sum(axis=0, numeric_only=True), columns=['sum'])

days_ = days_dates.merge(day_sums, left_index=True, right_index=True).set_index('date')

days_['sum'].iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,500), title='Sum of Sales', xTitle='Year',  mode='lines')
days_.groupby('wday')['sum'].agg(['mean']).iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,300), title='Sum of Sales By week day', xTitle='WeekDay', mode='lines+markers')
days_.groupby('month')['sum'].agg(['mean']).iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,500), title='Sum of Sales By month', xTitle='Month', mode='lines+markers')
days_[days_.event_name_1.notna()].groupby('event_name_1')['sum'].agg(['mean']).iplot(theme='space', color='gray', margin=(10, 10, 10, 40) ,dimensions=(800,500), title='Sum of Sales By Event Days', xTitle='Event', mode='lines+markers')
days_[days_.event_name_1.notna()].groupby('event_type_1')['sum'].agg(['mean']).iplot(theme='space', color='gray', kind = 'bar', margin=(10, 10, 10, 40) ,dimensions=(800,300),  title='Sum of Sales By Event Type', xTitle='Event Type', mode='lines+markers')
days_[days_.event_name_1.notna()].groupby('event_name_2')['sum'].agg(['mean']).iplot(theme='space', color='gray', kind = 'bar', margin=(10, 10, 10, 40) ,dimensions=(600,300),  title='Sum of Sales By Event Days', xTitle='Event', mode='lines+markers')
days_[days_.event_name_1.notna()].groupby('event_type_2')['sum'].agg(['mean']).iplot(theme='space', color='gray', kind = 'bar', margin=(10, 10, 10, 40) ,dimensions=(300,300), title='Sum of Sales By Event Type', xTitle='Event Type', mode='lines+markers')
days_.groupby('snap_CA')['sum'].agg(['mean']).iplot(theme='space', color='gray', kind = 'bar', margin=(10, 10, 10, 40) ,dimensions=(300,300), title='Sales By snap_CA', mode='lines+markers')
days_.groupby('snap_TX')['sum'].agg(['mean']).iplot(theme='space', color='gray', kind = 'bar', margin=(10, 10, 10, 40) ,dimensions=(300,300), title='Sales By snap_TX', mode='lines+markers')
days_.groupby('snap_WI')['sum'].agg(['mean']).iplot(theme='space', color='gray', kind = 'bar', margin=(10, 10, 10, 40) ,dimensions=(300,300), title='Sales By snap_WI', mode='lines+markers')
data_sample = sales_train_validation.loc[:,days].T

dist = (data_sample==0).sum()

hist_data = [dist]

group_labels = ['Zero Sale distribution']

fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(autosize=False, width=800, height=500, margin=dict(l=10, r=10, b=10, t=40))

fig.update_layout(template='plotly_dark', title_text='Zero Sale Days distribution')

fig.show()
data_sample = sales_train_validation.loc[:,days].T

dist = (data_sample==1).sum()

hist_data = [dist]

group_labels = ['One Sale distribution']

fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(autosize=False, width=800, height=500, margin=dict(l=10, r=10, b=10, t=40))

fig.update_layout(template='plotly_dark', title_text='One Sale Days distribution')

fig.show()
data_sample = sales_train_validation.loc[:,days].T

dist = (data_sample==2).sum()

hist_data = [dist]

group_labels = ['Two Sale distribution']

fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(autosize=False, width=800, height=500, margin=dict(l=10, r=10, b=10, t=40))

fig.update_layout(template='plotly_dark', title_text='Two Sale Days distribution')

fig.show()
data_sample = sales_train_validation.loc[:,days].T

dist = (data_sample==3).sum()

hist_data = [dist]

group_labels = ['Three Sale distribution']

fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(autosize=False, width=800, height=500, margin=dict(l=10, r=10, b=10, t=40))

fig.update_layout(template='plotly_dark', title_text='Three Sale Days distribution')

fig.show()
itx = items.T.iloc[:5].reset_index()

plt.figure()

ax = andrews_curves(itx, class_column='index', colormap='cubehelix')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()

plt.figure()

ax = autocorrelation_plot(items.T.iloc[1])

ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()
plt.figure()

ax = lag_plot(items.T.iloc[1])

ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.show()
weeks_per_year = 260 #five year period

plt.rcParams['figure.figsize'] = (16,12)

time_series = sales_sum_by_store["CA_1"]

sdc = seasonal_decompose(time_series, period = weeks_per_year)

sdcs = pd.DataFrame(sdc.seasonal).reset_index()

sdcs.columns = ['Days', 'Seasonal']

px.scatter(sdcs, x='Days', y='Seasonal', width=800, height=400, color='Seasonal', trendline='lowess', template='plotly_dark',   title='Seasonal Component')

fig.update_layout(

    margin=dict(l=10, r=10, t=40, b=20),

)

fig.show()
#https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f

def Croston(ts,extra_periods=1,alpha=0.4):

    d = np.array(ts) # Transform the input into a numpy array

    cols = len(d) # Historical period length

    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods

    

    #level (a), periodicity(p) and forecast (f)

    a,p,f = np.full((3,cols+extra_periods),np.nan)

    q = 1 #periods since last demand observation

    

    # Initialization

    first_occurence = np.argmax(d[:cols]>0)

    a[0] = d[first_occurence]

    p[0] = 1 + first_occurence

    f[0] = a[0]/p[0]

    # Create all the t+1 forecasts

    for t in range(0,cols):        

        if d[t] > 0:

            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 

            p[t+1] = alpha*q + (1-alpha)*p[t]

            f[t+1] = a[t+1]/p[t+1]

            q = 1           

        else:

            a[t+1] = a[t]

            p[t+1] = p[t]

            f[t+1] = f[t]

            q += 1

       

    # Future Forecast 

    a[cols+1:cols+extra_periods] = a[cols]

    p[cols+1:cols+extra_periods] = p[cols]

    f[cols+1:cols+extra_periods] = f[cols]

                      

    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})

    return df





def Croston_TSB(ts,extra_periods=1,alpha=0.4,beta=0.4):

    d = np.array(ts) # Transform the input into a numpy array

    cols = len(d) # Historical period length

    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods

    

    #level (a), probability(p) and forecast (f)

    a,p,f = np.full((3,cols+extra_periods),np.nan)

    # Initialization

    first_occurence = np.argmax(d[:cols]>0)

    a[0] = d[first_occurence]

    p[0] = 1/(1 + first_occurence)

    f[0] = p[0]*a[0]

                 

    # Create all the t+1 forecasts

    for t in range(0,cols): 

        if d[t] > 0:

            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 

            p[t+1] = beta*(1) + (1-beta)*p[t]  

        else:

            a[t+1] = a[t]

            p[t+1] = (1-beta)*p[t]       

        f[t+1] = p[t+1]*a[t+1]

        

    # Future Forecast

    a[cols+1:cols+extra_periods] = a[cols]

    p[cols+1:cols+extra_periods] = p[cols]

    f[cols+1:cols+extra_periods] = f[cols]

                      

    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})

    return df
days = range(1, 1913 + 1)

time_series_columns = [f'd_{i}' for i in days]

time_series_data = sales_train_validation[time_series_columns]



forecast_ = time_series_data.apply(lambda x : Croston_TSB(x, extra_periods=28, alpha=0.05,beta=0.31)['Forecast'].tail(28), axis=1)



cols = ['F'+str(i+1) for i in range(28)]

forecast_.columns = cols



validation_ids = sales_train_validation['id'].values

evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]

ids = np.concatenate([validation_ids, evaluation_ids])

predictions = pd.DataFrame(ids, columns=['id'])

forecast = pd.concat([forecast_] * 2).reset_index(drop=True)

predictions = pd.concat([predictions, forecast], axis=1)

predictions.to_csv('submission.csv', index=False)
