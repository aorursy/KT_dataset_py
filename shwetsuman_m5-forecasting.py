import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

!pip install calplot

import calplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

!pip install chart_studio 

import chart_studio.plotly as py

import plotly.graph_objs as go


# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
calendar = pd.read_csv('//kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sales_train_evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales_train_validation.head()
[column_train,row_train] = sales_train_validation.shape

column_train,row_train
print(len(sales_train_validation.id.str.contains('validation')))

print(len(sales_train_validation.id.unique()))
print(len(sales_train_validation.id.unique()))

print(len(sales_train_validation.item_id.unique()))

print(sales_train_validation.dept_id.unique())

print(sales_train_validation.cat_id.unique())

print(sales_train_validation.store_id.unique())

print(sales_train_validation.state_id.unique())
df = sales_train_validation

df1 = sales_train_validation.set_index('id').drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1).transpose()

df1.head()
n = len(calendar) - len(df1)

df2 = calendar[['date', 'd']].set_index('d').iloc[:-n]

df3 = pd.concat([df2,df1], axis=1).set_index('date')

df3
fig = plt.figure(figsize=(16,30))

for i,j in zip(df3.sample(n=20, axis=1), range(20)):

    ax=plt.subplot(10,2,j + 1) 

    df3[[i]].plot(ax=ax)

plt.show()
df4 = pd.concat([df[['id','item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].set_index('id'),df3.transpose()], axis=1)

df4
unique_units = df4.state_id.value_counts()

print('The count of total number of unique items is:\n', unique_units)

unique_units.plot(title='Distribution of Total items by State', kind='pie', autopct='%1.1f%%', figsize=(10,6))

plt.show()
Total_Sales = df4.groupby('state_id').sum().sum(axis=1)

Total_Sales.plot(title='Distribution of Total sales by State', kind='pie', autopct='%1.1f%%', figsize=(10,6))

plt.show()
df4.groupby(['cat_id']).sum().transpose().sum().plot(title='Sales Distribution by category', kind='pie', autopct='%1.1f%%',

        shadow=True, figsize=(10,6))

plt.show()
cat_dist = df4.groupby(['cat_id','state_id']).sum().sum(axis=1).unstack('cat_id')

(cat_dist.transpose() / cat_dist.transpose().sum()).transpose().plot(kind='bar')

plt.show()
df4.groupby(['dept_id']).sum().transpose().sum().plot(title='Sales Distribution by sub category', kind='pie', autopct='%1.1f%%',

        shadow=True, figsize=(10,6))

plt.show()
dept_dist = df4.groupby(['dept_id','state_id']).sum().sum(axis=1).unstack('dept_id')

(dept_dist.transpose() / dept_dist.transpose().sum()).transpose().iplot(kind='bar')
df4.groupby(['store_id']).sum().transpose().sum().plot(title='Sales Distribution by store', kind='pie', autopct='%1.1f%%',

        shadow=True, figsize=(10,6))

plt.show()
store_state_dist = df4.groupby(['store_id','state_id']).sum().sum(axis=1).unstack('store_id')

(store_state_dist.transpose() / store_state_dist.transpose().sum()).transpose().iplot(

    kind='bar', title='Sales distribution of stores in each state')

plt.show()
dept_store_dist = df4.groupby(['dept_id','store_id']).sum().sum(axis=1).unstack('dept_id')

(dept_store_dist.transpose() / dept_store_dist.transpose().sum()).transpose().iplot(

    kind='bar', title='Sales Distribution of Sub Categories by Store_ID')

plt.show()
sell_prices['wm_yr_wk'] = sell_prices['wm_yr_wk'].astype(str)



sell_prices['month'] = sell_prices['wm_yr_wk'].str[0:1]

sell_prices['year'] = sell_prices['wm_yr_wk'].str[1:3]

sell_prices['year'] = '20' + sell_prices['year'].astype(str)

sell_prices['week'] = sell_prices['wm_yr_wk'].str[3:5]

sell_prices['state_id'] = sell_prices['store_id'].str.split('_', 1).str[0]

sell_prices['cat_id'] = sell_prices['item_id'].str.split('_', 1).str[0]

sell_prices['dept_id'] = sell_prices['item_id'].str.split('_').str[0] + '_' + sell_prices['item_id'].str.split('_').str[1]



sell_prices.drop('wm_yr_wk', axis=1, inplace=True)
sell_prices.head()
sell_prices['sell_price'].plot(kind='hist', figsize=(10,5), bins=20)

plt.show()
sell_prices.groupby(['year','state_id']).mean().unstack('state_id').boxplot(

    figsize=(10,3), vert=False)

plt.show()
sell_prices.groupby(['year','store_id']).mean().unstack('store_id').boxplot(figsize=(10,7), vert=False)

plt.show()
sell_prices.groupby(['year','cat_id']).mean().unstack('cat_id').boxplot(figsize=(12,3), vert=False)

plt.show()
sell_prices.groupby(['year','dept_id']).mean().unstack('dept_id').boxplot(figsize=(12,4), vert=False)

plt.show()
sell_prices.groupby(['week','year']).mean().unstack('week').boxplot(figsize=(10,12), vert=False)

plt.show()
sell_prices.groupby(['week','year']).mean().unstack('year').boxplot(figsize=(14,5), vert=False)

plt.show()
calendar.head()
event_1 = pd.merge(calendar[['date','weekday','month','year','d']], 

                   calendar[['d','event_name_1','event_type_1']].dropna(), on='d')

print(event_1)

print("There are 162 events in the calender dateset")
event_1.groupby(['event_type_1','year'])['event_name_1'].size().unstack(

    'event_type_1').iplot(kind='barh', title='Event Type 1')

plt.show()
event_1.groupby(['month','year'])['event_name_1'].size().unstack('year').iplot(

    kind='bar', title='Events 1 by Month')

plt.show()
event_1.groupby(['weekday','year'])['event_name_1'].size().unstack('year').iplot(

    kind='bar', title='Event 1 by day of the Week')

plt.show()
event_2 = pd.merge(calendar[['date','weekday','month','year','d']], 

                   calendar[['d','event_name_2','event_type_2']].dropna(), on='d')

print(event_2)

print("There are 5 Event 2 in the calendar dataset")
event_2.groupby(['event_type_2','year'])['event_name_2'].size().unstack(

    'event_type_2').iplot(kind='barh', title='Event 2')

plt.show()
event_2.groupby(['month','year'])['event_name_2'].size().unstack('year').iplot(

    kind='barh', title='Event 2 by month of the Year')

plt.show()
event_2.groupby(['weekday','year'])['event_name_2'].size().unstack('year').iplot(

    kind='barh', title='Event 2 by day of the week')

plt.show()
pd.merge(calendar[['date','weekday','month','year','d']], 

                   calendar[['d','event_name_1','event_type_1','event_name_2','event_type_2']].dropna(), on='d')
snap = pd.merge(calendar[['date','weekday','month','year','d']], 

                   calendar[['d','snap_CA','snap_TX','snap_WI']].loc[~(calendar[['snap_CA','snap_TX','snap_WI']]==0).all(axis=1)], 

                on='d')

snap
snap_CA = snap.groupby(['snap_CA','year']).size()[1].to_frame().reset_index().rename(columns={0:'CA'})

snap_TX = snap.groupby(['snap_TX','year']).size()[1].to_frame().reset_index().rename(columns={0:'TX'})

snap_WI = snap.groupby(['snap_WI','year']).size()[1].to_frame().reset_index().rename(columns={0:'WI'})



pd.merge(pd.merge(snap_CA,snap_TX,on='year'),snap_WI,on='year').set_index(

    'year').iplot(kind='barh', title='SNAP Days')

plt.show()
snap_CA_1 = snap.groupby(['snap_CA','month']).size()[1].to_frame().reset_index().rename(columns={0:'CA'})

snap_TX_1 = snap.groupby(['snap_TX','month']).size()[1].to_frame().reset_index().rename(columns={0:'TX'})

snap_WI_1 = snap.groupby(['snap_WI','month']).size()[1].to_frame().reset_index().rename(columns={0:'WI'})



pd.merge(pd.merge(snap_CA_1,snap_TX_1,on='month'),snap_WI_1,on='month').set_index(

    'month').iplot(kind='barh', title='SNAP Days by month of the year')

plt.show()
snap.groupby(['snap_CA','month','year']).size()[1].unstack('year').iplot(

    kind='bar', title='CA SNAP Days by month')

plt.show()
snap.groupby(['snap_TX','month','year']).size()[1].unstack('year').iplot(

    kind='bar', title='TX SNAP Days by month')

plt.show()
snap.groupby(['snap_WI','month','year']).size()[1].unstack('year').iplot(

    kind='bar', title='WI SNAP Days by month')

plt.show()
snap.groupby(['snap_CA','weekday','year']).size()[1].unstack('year').iplot(

    kind='barh', title='CA SNAP Days by day of the Week')

plt.show()
snap.groupby(['snap_TX','weekday','year']).size()[1].unstack('year').iplot(

    kind='barh', title='TX SNAP Days by day of the Week')

plt.show()
snap.groupby(['snap_WI','weekday','year']).size()[1].unstack('year').iplot(

    kind='barh', title='WI SNAP Days by day of the Week')

plt.show()
pd.merge(calendar[['date','weekday','month','year','d']], 

                   calendar[['d','snap_CA','snap_TX','snap_WI']].loc[

                       (calendar[['snap_CA','snap_TX','snap_WI']]==1).all(axis=1)], 

                on='d')
days = list(pd.to_datetime(calendar.date))

events = pd.Series(list(calendar.snap_CA), index=days)



calplot.calplot(events, cmap='RdBu', colorbar=False)

plt.show()
days = list(pd.to_datetime(calendar.date))

events = pd.Series(list(calendar.snap_TX), index=days)



calplot.calplot(events, cmap='RdBu', colorbar=False)

plt.show()
days = list(pd.to_datetime(calendar.date))

events = pd.Series(list(calendar.snap_WI), index=days)



calplot.calplot(events, cmap='RdBu', colorbar=False)

plt.show()
cummulative_sales = df3.transpose().sum().to_frame().rename(columns={0:'cummulative_sales'})

cummulative_sales.head()
cummulative_sales.iplot(title='Time Series Plots - Cummulative Sales')

plt.show()
result = seasonal_decompose(cummulative_sales[

    'cummulative_sales'].values, period=7, model='additive')

plt.rcParams.update({'figure.figsize': (12,8)})

result.plot().suptitle('Additive Decomposition', fontsize=22)

plt.show()
df3.index = pd.to_datetime(df3.index)

df5 = pd.concat([df[['id','item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].set_index('id'),

                 df3.groupby(pd.Grouper(freq='1M')).sum().transpose()], axis=1)
df5.groupby('state_id').sum().transpose().iplot(title='Time Series Plots - Statewise')

plt.show()
result = seasonal_decompose(df.groupby(

    'state_id').sum().transpose().CA.values, period=7, model='additive')

plt.rcParams.update({'figure.figsize': (12,8)})

result.plot().suptitle('Time Series of CA', fontsize=22)

plt.show()
result = seasonal_decompose(df4.groupby(

    'state_id').sum().transpose().TX.values, period=7, model='additive')

plt.rcParams.update({'figure.figsize': (12,8)})

result.plot().suptitle('Time Series of TX', fontsize=22)

plt.show()
result = seasonal_decompose(df4.groupby(

    'state_id').sum().transpose().WI.values, period=7, model='additive')

plt.rcParams.update({'figure.figsize': (12,8)})

result.plot().suptitle('Time Series of WI', fontsize=22)

plt.show()
df5.groupby(['state_id','cat_id']).sum().transpose().CA.iplot(title='CA sales by Category')

plt.show()
df5.groupby(['state_id','cat_id']).sum().transpose().TX.iplot(title='TX sales by Category')

plt.show()
df5.groupby(['state_id','cat_id']).sum().transpose().WI.iplot(title='WI sales by Category')

plt.show()
df5.groupby(['state_id','store_id']).sum().transpose().CA.iplot(title='CA sales by Stores')

plt.show()
df5.groupby(['state_id','store_id']).sum().transpose().TX.iplot(title='TX sales by Store')

plt.show()
df5.groupby(['state_id','store_id']).sum().transpose().WI.iplot(title='WI sales by Store')

plt.show()
df5.groupby(['store_id']).sum().transpose().iplot(title='Sales by Stores')

plt.show()
df7 = pd.merge(calendar[['date','weekday','month','year']], 

                   pd.concat([df2,df1], axis=1), on='date').set_index('date')

df7.head()
df7.drop(['month'], axis=1).groupby(['weekday','year']).sum().sum(axis=1).unstack(

    'weekday').iplot(kind='bar', title='Sales by day of the Week')

plt.show()
df7.drop(['weekday'], axis=1).groupby(['month','year']).sum().sum(axis=1).unstack(

    'month').iplot(kind='bar', title='Sales by Month')

plt.show()
days = list(pd.to_datetime(cummulative_sales.index))

events = pd.Series(list(cummulative_sales.cummulative_sales), index=days)



calplot.calplot(events, cmap='CMRmap')

plt.show()
cummulative_sales.iplot(kind='hist')

plt.show()
cummulative_sales_1 = pd.merge(calendar, cummulative_sales.reset_index(), on='date')

cummulative_sales_1.head()
cummulative_sales_1.groupby(['weekday']).mean()['cummulative_sales'].plot(kind='barh', figsize=(12,6))

plt.axvline(x=cummulative_sales_1.cummulative_sales.mean(), color='k', linestyle='--')

plt.show()
cummulative_sales_2 = cummulative_sales_1.groupby(['weekday']).mean()['cummulative_sales'].reset_index()

cummulative_sales_2.loc[cummulative_sales_2.weekday=='Saturday']['cummulative_sales'].values
cummulative_sales_2.loc[

    (cummulative_sales_2.weekday=='Saturday') | (cummulative_sales_2.weekday=='Sunday')].mean()
event_days_sales = cummulative_sales_1[

    ((cummulative_sales_1.event_name_1.notnull()) | (cummulative_sales_1.event_name_2.notnull()))]

cummulative_sales_1["weekend_precede_event"] = np.nan



def update_weekend_precede_event(week_e,wday,e1,e2):

    e2 = '_' + e2 if type(e2) == str else ''

    drift = e1 + e2

    if wday == 1:

        cummulative_sales_1.loc[

            (cummulative_sales_1['wm_yr_wk']==week_e)&(cummulative_sales_1[

                'wday']==1),"weekend_precede_event"] = drift

    else:

        cummulative_sales_1.loc[

            (cummulative_sales_1[

                'wm_yr_wk']==week_e)&((cummulative_sales_1['wday']==1)|(cummulative_sales_1[

                'wday']==2)),"weekend_precede_event"] = drift

        

_ = event_days_sales.apply(lambda row : update_weekend_precede_event(row[

    'wm_yr_wk'],row['wday'],row['event_name_1'], row['event_name_2']),axis = 1)
cummulative_sales_1.head()
cummulative_sales_1.groupby(['weekend_precede_event','weekday'])[

    'cummulative_sales'].mean().unstack('weekday').mean(axis=1).sort_values(ascending = False).plot(kind='bar', figsize=(16,6))

plt.axhline(y=cummulative_sales_2.loc[

    (cummulative_sales_2.weekday=='Saturday') | (

        cummulative_sales_2.weekday=='Sunday')].mean().values, color='black', linestyle='--')

plt.show()
snap_1 = pd.merge(snap, cummulative_sales, on='date')

snap_1
snap_CA_1 = pd.merge(snap[['date','snap_CA']], df4.groupby([

    'state_id']).sum().T['CA'].reset_index().rename(

    columns={'index':'date'}), on='date').groupby(['snap_CA']).mean().reset_index()

snap_CA_1.columns = ['snap', 'CA_sales']

snap_TX_1 = pd.merge(snap[['date','snap_TX']], df4.groupby([

    'state_id']).sum().T['TX'].reset_index().rename(

    columns={'index':'date'}), on='date').groupby(['snap_TX']).mean().reset_index()

snap_TX_1.columns = ['snap', 'TX_sales']

snap_WI_1 = pd.merge(snap[['date','snap_WI']], df4.groupby([

    'state_id']).sum().T['WI'].reset_index().rename(

    columns={'index':'date'}), on='date').groupby(['snap_WI']).mean().reset_index()

snap_WI_1.columns = ['snap', 'WI_sales']
pd.merge(pd.merge(snap_CA_1,snap_TX_1, on='snap'),snap_WI_1, on='snap').set_index('snap').T.plot(

    kind='bar', figsize=(10,8), title='Snap Days effect')

plt.axhline(y=df4.groupby(['state_id']).sum().mean(axis=1).to_frame().T.CA.values, color='red', linestyle='--')

plt.text(0,df4.groupby(['state_id']).sum().mean(axis=1).to_frame().T.CA.values,'Average sales in CA', size=14)

plt.axhline(y=df4.groupby(['state_id']).sum().mean(axis=1).to_frame().T.TX.values, color='k', linestyle='--')

plt.text(0.5,df4.groupby(['state_id']).sum().mean(axis=1).to_frame().T.TX.values,'Average sales in TX', size=14)

plt.axhline(y=df4.groupby(['state_id']).sum().mean(axis=1).to_frame().T.WI.values, color='blue', linestyle='--')

plt.text(1.5,df4.groupby(['state_id']).sum().mean(axis=1).to_frame().T.WI.values,'Average sales in WI', size=14)

plt.show()