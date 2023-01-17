# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import os

import csv

import datetime as datetime

import json 

import datetime as dttm #time/date manipulations

import sklearn

#import matplotlib.pyplot as plot#Matplotlib

#import seaborn as sns

import numpy as np

import math

import keras

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



import seaborn as sns

import matplotlib.pylab as plab



from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## load the data

sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')



#Print dimensions of dataframe

print(sales_train.shape)

print(sell_prices.shape)

print(calendar.shape)





#Print first few rows of the data frame

print(sales_train.head())

print(sell_prices.head())

print(calendar.head())
#Check counts of non-zero values in all the time series - 

sales_prods= sales_train.loc[:, 'd_1' : ]

zero = sales_prods.apply(lambda x: x == 0)



k=zero.sum(axis=1)

k=k/1913

sales_train['zero_days']=k

#sales_train



hobbies=sales_train.loc[sales_train['cat_id']=='HOBBIES']

foods=sales_train.loc[sales_train['cat_id']=='FOODS']

household=sales_train.loc[sales_train['cat_id']=='HOUSEHOLD']

#print(sales_train.groupby(['cat_id']).agg({'zero_days': ['mean','median']}).reset_index())



fig = go.Figure()

fig.add_trace(go.Histogram(x=hobbies['zero_days'],name='Hobbies'))

fig.add_trace(go.Histogram(x=foods['zero_days'],name='Foods'))

fig.add_trace(go.Histogram(x=household['zero_days'],name='Household'))



# Overlay all histograms

fig.update_layout(barmode='overlay',title_text='Distribution of Zero Values Across Product Categories',xaxis_title_text='Proportion of Zero Values',yaxis_title_text='Count')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.1)



fig.show()
#Checking how zero values vary by state

tx=sales_train.loc[sales_train['state_id']=='TX']

ca=sales_train.loc[sales_train['state_id']=='CA']

wi=sales_train.loc[sales_train['state_id']=='WI']





fig = go.Figure()

fig.add_trace(go.Histogram(x=tx['zero_days'],name='TX'))

fig.add_trace(go.Histogram(x=ca['zero_days'],name='CA'))

fig.add_trace(go.Histogram(x=wi['zero_days'],name='WI'))





# Overlay all histograms

fig.update_layout(barmode='overlay',title_text='Distribution of Zero Values Across States',xaxis_title_text='Proportion of Zero Values',yaxis_title_text='Count')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.3)



fig.show()
#Sample dataframe to navigate tough waters ahead

print(sales_train.shape)



rows=int(sales_train.shape[0]/4)

sales_train=sales_train.sample(rows, replace=True)

print(sales_train.shape)



#Data transpose to convert dataset from wide format to long format

sales_train_l=sales_train.melt(['id','item_id','dept_id','cat_id','store_id','state_id'], var_name='d', value_name='qty')



print(sales_train_l.shape)



sales_train_l.head()
#Join sales time series with calendar dates 

cols = ['date','wm_yr_wk','d','weekday','month','year','event_type_1','snap_CA','snap_TX','snap_WI']

calendar_dts = calendar[cols]



calendar_dts.head()

sales_train_l_dt = pd.merge(left=sales_train_l,right=calendar_dts,left_on='d', right_on='d')



cols = ['date','d','weekday','month','year']

calendar_dts = calendar[cols]



#calendar_dts.head()

#sales_train_l_dt = pd.merge(left=sales_train_l,right=calendar,left_on='d', right_on='d')



sales_train_l_dt['event_type_1']=sales_train_l_dt['event_type_1'].fillna('No Event')



print(sales_train_l.shape)

print(sales_train_l_dt.shape)

sales_train_l_dt.head(5)
#Installing and loading the calendar plot package

!pip install calplot



import calplot
#Checking number of SNAP days in a year across the three states

snappy=calendar.groupby(['year','month']).agg({'snap_CA': ['sum'],'snap_TX': ['sum'],'snap_WI': ['sum']}).reset_index()

snappy
#plotting calendar heatmap for California



x=calendar[['date','snap_CA']]

events=x['snap_CA']

events.index=pd.to_datetime(x['date'])

calplot.calplot(events)
#Check relationship between SNAP days and sales. Are there more products being sold on snap days?



snap=sales_train_l_dt.groupby(['store_id','cat_id','date','snap_CA','snap_TX','snap_WI']).agg({'qty': ['sum']}).reset_index()



state=['CA','TX','WI']

l=[]



for i in range(3):

  a=snap['store_id'].str.contains(state[i],case=False,regex=True) 

  snap_st=snap.loc[a]

  snap_st.columns=['store_id','cat_id','date','snap_CA','snap_TX','snap_WI','qty']

  col="snap_"+state[i]

  snap_st_gp=snap_st.groupby([col,'cat_id']).agg({'qty': ['sum'],'date': ['nunique']}).reset_index()

  snap_st_gp.columns=[col,'cat_id','qty','unique_dates']

  snap_st_gp['qty_day']=snap_st_gp['qty']/snap_st_gp['unique_dates']

  snap_st_gp['state']=state[i]

  snap_st_gp.columns=['SNAP','cat_id','qty','unique_dates','qty_day','state']

  l.append(snap_st_gp)

  #print(snap_st_gp)

    

snap_append = pd.concat(l).reset_index(drop=True)

snap_append=snap_append[['SNAP','cat_id','qty_day','state']]

#snap_append



snap_append['comb']=snap_append['cat_id']+' '+snap_append['state']

del snap_append['state']

del snap_append['cat_id']

pivot_snap=snap_append.pivot(index='comb', columns='SNAP', values='qty_day')

pivot_snap['comb']=pivot_snap.index

pivot_snap.reset_index(drop=True)

pivot_snap.columns=['snap_0','snap_1','comb']



pivot_snap = pivot_snap.set_index('comb')

pivot_snap=pivot_snap.div(pivot_snap.sum(axis=1), axis=0)

pivot_snap=pivot_snap.reset_index()





#Plot stacked bar chart

fig = go.Figure(data=[

    go.Bar(name='Avg. Qty. Non-SNAP Day', x=pivot_snap['comb'], y=pivot_snap['snap_0']),

    go.Bar(name='Avg. Qty. SNAP Day', x=pivot_snap['comb'], y=pivot_snap['snap_1'])

])

# Change the bar mode

fig.update_layout(barmode='stack',title_text='Impact of SNAP days on Sales',xaxis_title_text='State_Category',yaxis_title_text='Percentage'

)



fig.show()
output=calendar.groupby('event_type_1').agg({'date': ['nunique']}).reset_index()

output.columns=['event_type_1','days']



fig = go.Figure(data=[

    go.Bar(name='Avg. Qty. Non-SNAP Day', x=output['event_type_1'], y=output['days'])

])

fig.update_layout(title_text='Number of days by Event Type',xaxis_title_text='Event Type',yaxis_title_text='No. of Days'

)



fig.show()
output=calendar.groupby(['event_type_1','event_name_1']).agg({'date': ['nunique']}).reset_index()

output.columns=['event_type_1','event_name_1','days']





fig = px.bar(output, x="event_name_1", y="days", color='event_type_1')

fig.update_layout(title_text='Number of days by Event Name',xaxis_title_text='Event Type',yaxis_title_text='No. of Days')





fig.show()
#Check relationship between Event days and sales. Are there more products being sold on event days?

event=sales_train_l_dt.groupby(['cat_id','state_id','event_type_1']).agg({'qty': ['sum'],'date': ['nunique']}).reset_index()

event.columns=['cat_id','state_id','event_type_1','qty','days']

event['qty_per_day']=event['qty']/event['days']



event_pivot=event.groupby(['cat_id','state_id','event_type_1'])['qty_per_day'].sum().unstack('event_type_1')



event_pivot.reset_index(inplace=True)

event_pivot['cat_state']=event_pivot['cat_id']+" " +event_pivot['state_id']

event_pivot.index=event_pivot['cat_state']



del event_pivot['cat_id']

del event_pivot['state_id']

del event_pivot['cat_state']



event_pivot=(event_pivot.T / event_pivot.T.sum()).T

ax = sns.heatmap(event_pivot,linewidths=.15,cmap='YlGnBu')
#Correlation between price and demand



sales_train_p_dt = pd.merge(left=sales_train_l_dt,right=sell_prices,left_on=['store_id','item_id','wm_yr_wk'],right_on=['store_id','item_id','wm_yr_wk'])



sales_group=sales_train_p_dt.groupby(['item_id','month','year']).agg({'sell_price': ['median'],'qty':['sum']}).reset_index()

sales_group.columns=['item_id','month','year','price','qty']



sales_group.head()

items=sales_group.item_id.unique() 

l=[]

for i in range(len(items)):

  df=sales_group.loc[sales_group['item_id']==items[i]]

  cor=df['price'].corr(df['qty'])

  l.append(cor)



ser=pd.Series(l)

ser=ser.fillna(0)

items=pd.Series(items)

#ser

corframe=pd.concat([items, ser], axis=1)

corframe.columns=['item_id','correlation']

corframe.shape[0]

cat=[]

for i in range(corframe.shape[0]):

   cat.append(corframe['item_id'].str.split('_')[i][0])



corframe['cat']=pd.Series(cat)



foods=corframe.loc[corframe['cat']=='FOODS']

hobbies=corframe.loc[corframe['cat']=='HOBBIES']

household=corframe.loc[corframe['cat']=='HOUSEHOLD']





fig = go.Figure()

fig.add_trace(go.Histogram(x=foods['correlation'],name='FOODS'))

fig.add_trace(go.Histogram(x=hobbies['correlation'],name='HOBBIES'))

fig.add_trace(go.Histogram(x=household['correlation'],name='HOUSEHOLD'))





# Overlay all histograms

fig.update_layout(barmode='overlay',title_text='Distribution of Correlation Between Price & Sales Values Across Products',xaxis_title_text='Correlation',yaxis_title_text='Count of Item_IDs')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.3)



fig.show()
    

group_price_cat = sales_train_p_dt.groupby(['date','cat_id'],as_index=False)['sell_price'].mean().dropna()

group_price_cat.head()



#fig = px.line(group_price_cat, x="date", y="sell_price", color='cat_id')



fig = make_subplots(rows=1, cols=3)



fig.add_trace(

    go.Scatter(x=group_price_cat[group_price_cat['cat_id']=='HOBBIES']['date'], y=group_price_cat[group_price_cat['cat_id']=='HOBBIES']['sell_price'],name='HOBBIES'),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=group_price_cat[group_price_cat['cat_id']=='HOUSEHOLD']['date'], y=group_price_cat[group_price_cat['cat_id']=='HOUSEHOLD']['sell_price'],name='HOUSEHOLD'),

    row=1, col=2

)



fig.add_trace(

    go.Scatter(x=group_price_cat[group_price_cat['cat_id']=='FOODS']['date'], y=group_price_cat[group_price_cat['cat_id']=='FOODS']['sell_price'],name='FOODS'),

    row=1, col=3

)



fig.update_layout(title_text='Avg. Sell Prices across categories over time',xaxis_title_text='date',yaxis_title_text='Sell Price($)')

fig.show()



cal_sub=calendar.drop_duplicates(subset='wm_yr_wk', keep="first")

cal_sub=cal_sub[['wm_yr_wk','date']]



sell_prices_cal=pd.merge(left=sell_prices,right=cal_sub,left_on='wm_yr_wk', right_on='wm_yr_wk')

#sell_prices_cal.head(1)

sell_prices_cal['cat_id'] = sell_prices_cal['item_id'].str.split('_').str[0]



unique_prods=sell_prices_cal.groupby(['date','cat_id'],as_index=False).agg({'item_id':['nunique']})

unique_prods.columns=['date','cat_id','unique_items']



fig = px.line(unique_prods, x="date", y="unique_items", color='cat_id')

fig.update_layout(title_text='Total Unique Products in a Category over time',xaxis_title_text='date',yaxis_title_text='Unique Prods Count')



fig.show()



#Product prices by store

import plotly.express as px



group_price_cat = sales_train_p_dt.groupby(['store_id','cat_id','item_id'],as_index=False)['sell_price'].mean().dropna()

group_price_cat.head()



fig = px.violin(group_price_cat, x='store_id', color='cat_id', y='sell_price',box=True, hover_name='item_id') 

fig.update_layout(template='seaborn',title='Distribution of Item Prices across Stores',legend_title_text='Category')



fig.show()
group_qty_cat = sales_train_p_dt.groupby(['year','date','state_id','store_id'],as_index=False)['qty'].sum().dropna()



fig = px.violin(group_qty_cat, x='store_id', color='state_id',y='qty',box=True) 

fig.update_layout(template='seaborn',title='Distribution of Quantity sold for Stores',legend_title_text='State')



fig.show()
#Correlation between sales for product categories across states

state_daily=sales_train_l_dt.groupby(['state_id','cat_id','date'],as_index=False)['qty'].sum().dropna()

state_daily['cat_state']=state_daily['state_id']+" "+state_daily['cat_id']





del state_daily['cat_id']

del state_daily['state_id']

k=pd.pivot_table(state_daily, values='qty', index=['date'], columns=['cat_state'])

corr=k.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
#Outputs for plotting

daily=sales_train_l_dt.groupby('date').agg({'qty': ['sum']}).reset_index()

daily.columns=['date','qty']

daily.head()



#State level sales 

state=sales_train_l_dt.groupby(['state_id','date']).agg({'qty': ['sum']}).reset_index()

state.columns=['state_id','date','qty']

state.head()



#weekday vs weekend

weekday=sales_train_l_dt.groupby(['weekday']).agg({'qty': ['sum']}).reset_index()

weekday.columns=['weekday','qty']

weekday.head()





#category

cat=sales_train_l_dt.groupby(['cat_id','date']).agg({'qty': ['sum']}).reset_index()

cat.columns=['cat_id','date','qty']

cat.head()



#Outputs for plotting

daily=sales_train_l_dt.groupby('date').agg({'qty': ['sum']}).reset_index()

daily.columns=['date','qty']

daily.head()



#State level sales 

state=sales_train_l_dt.groupby(['state_id','date']).agg({'qty': ['sum']}).reset_index()

state.columns=['state_id','date','qty']

state.head()



#weekday vs weekend

weekday=sales_train_l_dt.groupby(['weekday']).agg({'qty': ['sum']}).reset_index()

weekday.columns=['weekday','qty']

weekday.head()



weekday_cat=sales_train_l_dt.groupby(['weekday','cat_id']).agg({'qty': ['sum']}).reset_index()

weekday_cat.columns=['weekday','cat_id','qty']

weekday_cat.head()







#monthly

month=sales_train_l_dt.groupby(['month']).agg({'qty': ['sum']}).reset_index()

month.columns=['month','qty']

month.head()



#monthly cat

month_cat=sales_train_l_dt.groupby(['month','cat_id']).agg({'qty': ['sum']}).reset_index()

month_cat.columns=['month','cat_id','qty']

month_cat.head()





#store

store=sales_train_l_dt.groupby(['store_id','year']).agg({'qty': ['sum']}).reset_index()

store.columns=['store_id','year','qty']

store.head()
import plotly.graph_objects as go

fig = go.Figure( go.Scatter(x=daily['date'], y=daily['qty'] ) )

fig.update_layout(title='Time series for overall quantity sold',xaxis_title_text='Date',yaxis_title_text='Qty Sold')



fig.show()
# Create traces

fig = go.Figure()

state_list=['CA','TX','WI']

for i in range(3):

  subset=state[state['state_id']==state_list[i]]

  fig.add_trace(go.Scatter(x=subset['date'], y=subset['qty'],

                    mode='lines',

                    name=state_list[i]))



fig.update_layout(title='Time series for overall quantity across States',xaxis_title_text='Date',yaxis_title_text='Qty Sold')

fig.show()
from plotly.subplots import make_subplots



cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']





weekday['weekday'] = pd.Categorical(weekday['weekday'], categories=cats, ordered=True)

weekday_cat['weekday'] = pd.Categorical(weekday_cat['weekday'], categories=cats, ordered=True)



weekday = weekday.sort_values('weekday')

weekday_cat = weekday_cat.sort_values('weekday')



fig = make_subplots(rows=4, cols=1)



fig.add_trace(

    go.Scatter(x=weekday['weekday'], y=weekday['qty'],name='OVERALL'),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=weekday_cat[weekday_cat['cat_id']=='HOBBIES']['weekday'], y=weekday_cat[weekday_cat['cat_id']=='HOBBIES']['qty'],name='HOBBIES'),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(x=weekday_cat[weekday_cat['cat_id']=='FOODS']['weekday'], y=weekday_cat[weekday_cat['cat_id']=='FOODS']['qty'],name='FOODS'),

    row=3, col=1

)

fig.add_trace(

    go.Scatter(x=weekday_cat[weekday_cat['cat_id']=='HOUSEHOLD']['weekday'], y=weekday_cat[weekday_cat['cat_id']=='HOUSEHOLD']['qty'],name='HOUSEHOLD'),

    row=4, col=1

)





fig.update_layout(height=600, width=800, title_text="Aggregated Sales Volume by Day of Week")

fig.show()
from plotly.subplots import make_subplots



fig = make_subplots(rows=4, cols=1)



fig.add_trace(

    go.Scatter(x=month['month'], y=month['qty'],name='OVERALL'),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=month_cat[month_cat['cat_id']=='HOBBIES']['month'], y=month_cat[month_cat['cat_id']=='HOBBIES']['qty'],name='HOBBIES'),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(x=month_cat[month_cat['cat_id']=='FOODS']['month'], y=month_cat[month_cat['cat_id']=='FOODS']['qty'],name='FOODS'),

    row=3, col=1

)

fig.add_trace(

    go.Scatter(x=month_cat[month_cat['cat_id']=='HOUSEHOLD']['month'], y=month_cat[month_cat['cat_id']=='HOUSEHOLD']['qty'],name='HOUSEHOLD'),

    row=4, col=1

)





fig.update_layout(height=600, width=800, title_text="Aggregated Sales Volume by Day of Month")

fig.show()
# Create traces for stores

store_list=store.store_id.unique()

year_list=store.year.unique()



data=[]

for i in range(len(year_list)):

  data.append(go.Bar(name=str(year_list[i]), x=store_list, y=store.loc[(store['year']==year_list[i]),'qty']))

           

fig = go.Figure(data)

# Change the bar mode

fig.update_layout(barmode='stack',title='Overall Quantity across Stores over time ',xaxis_title_text='Store',yaxis_title_text='Qty Sold')



fig.show()
#Function to implement exponential smoothing

def exponential_smoothing(series, alpha):

    """

        series - dataset with timestamps

        alpha - float [0.0, 1.0], smoothing parameter

    """

    result = [series[0]] # first value is same as series

   # print(result)

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result



#Croston's Method - https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f

def Croston_TSB(ts,extra_periods=1,alpha=0.5,beta=0.7):

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
from random import randrange



# Create traces

fig = go.Figure()

rand=randrange(1,20000)

temp=sales_train_l_dt.iloc[rand]

#print(temp)

#print(type(temp))

item=temp['item_id']

store=temp['store_id']

print(item)



subset=sales_train_l_dt[(sales_train_l_dt['item_id']==item) & (sales_train_l_dt['store_id']==store)]

print(subset.shape)



sma = subset.qty.rolling(30).mean()

res=exponential_smoothing(subset['qty'].reset_index(drop=True), 0.05)

#k

print(len(res))



#Croston Forecast

cros=Croston_TSB(subset['qty'].reset_index(drop=True),extra_periods=28)

cros_forecast=cros['Forecast']

print('Croston sum')

print(np.sum(cros_forecast))



print('Actual sum')

print(np.sum(subset['qty']))





fig.add_trace(go.Scatter(x=subset['date'], y=subset['qty'],

                     mode='lines',

                     name=item))



fig.add_trace(go.Scatter(x=subset['date'], y=sma,

                      mode='lines',

                      name='SMA_30'))



fig.add_trace(go.Scatter(x=subset['date'], y=res,

                      mode='lines',

                      name='EXP'))



fig.add_trace(go.Scatter(x=subset['date'], y=cros_forecast,

                      mode='lines',

                      name='Croston'))



fig.update_layout(barmode='stack',title='Time Series Forecasts for a Random Time Series ',xaxis_title_text='Date',yaxis_title_text='Qty Sold')





fig.show()
#Naive - Just pick the last sales value 

naive=sales_train.iloc[:, -1]

naive=naive.reset_index(drop=True)



#print(naive)



k=pd.Series(naive[0])

k=k.repeat(28).reset_index(drop=True)

#print(type(k))

naive_time_series_all=[]

naive_time_series_all.extend([naive for i in range(28)])
#Define Root Mean Squared Error

from sklearn.metrics import mean_squared_error

from math import sqrt

x=[]



actual=sales_train.iloc[:, -28:]

time_series_all=sales_train.iloc[:, 7:]

#moving average

moving_average_series_all=[]



#Exponential Series Average 

exponential_average_series_all=[]



for i in range(time_series_all.shape[0]):

  series=time_series_all.iloc[i].rolling(28).mean()  

  moving_average_series_all.append(series.tail(28))



#Seasonal Naive

seasonal_naive=sales_train.iloc[:, -365:-337]



#Exponential Average Smoothing



for i in range(time_series_all.shape[0]):

  series=time_series_all.iloc[i]

  output=exponential_smoothing(series,0.1)

  output=pd.Series(output)

  exponential_average_series_all.append(output.tail(28))
#30 day average

ave_28=actual.apply(np.mean,axis=1)

ave_28=ave_28.reset_index(drop=True)



#Croston Method

croston_series_all=[]



for i in range(time_series_all.shape[0]):

  series=time_series_all.iloc[i]

  res=Croston_TSB(series,extra_periods=1)

  output=res['Forecast']

  #print(np.sum(output[-29:-1]))

  croston_series_all.append(output[-29:-1])
#Compute RMS values for all methods

rms_naive=[]

rms_ave_28=[]

rms_seasonal_naive=[]

rms_ma=[]

rms_es=[]

rms_croston=[]



for i in range(7622):

  y_actual=time_series_all.iloc[i,-28:]

  y_predicted_naive=pd.Series(naive[i])

  y_predicted_naive=y_predicted_naive.repeat(28).reset_index(drop=True)

  y_predicted_ave_28=pd.Series(ave_28[i])

  y_predicted_ave_28=y_predicted_ave_28.repeat(28).reset_index(drop=True)

  y_predicted_seasonal_naive=seasonal_naive.iloc[i,:]

  y_predicted_ma=moving_average_series_all[i]

  y_predicted_es=exponential_average_series_all[i].tail(28)

  y_predicted_croston=croston_series_all[i].tail(28)

  

  rms_naive.append(sqrt(mean_squared_error(y_actual, y_predicted_naive)))

  rms_ave_28.append(sqrt(mean_squared_error(y_actual, y_predicted_ave_28)))  

  rms_seasonal_naive.append(sqrt(mean_squared_error(y_actual, y_predicted_seasonal_naive)))

  rms_ma.append(sqrt(mean_squared_error(y_actual, y_predicted_ma)))

  rms_es.append(sqrt(mean_squared_error(y_actual, y_predicted_es)))

  rms_croston.append(sqrt(mean_squared_error(y_actual, y_predicted_croston)))



print('Error with Naive')

print(sum(rms_naive))



print('Error with Ave 28')

print(sum(rms_ave_28))



print('Error with Seasonal Naive')

print(sum(rms_seasonal_naive))



print('Error with MA')

print(sum(rms_ma))



print('Error with ES')

print(sum(rms_es))



print('Error with Croston Method')

print(sum(rms_croston))
#Import raw data and extract columns

sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

time_series_all=sales_train.iloc[:, 7:]

exponential_average_series_all=[]

croston_series_all=[]



#Croston Method

for i in range(time_series_all.shape[0]):

  series=time_series_all.iloc[i]

  res=Croston_TSB(series,extra_periods=1)

  output=res['Forecast']

  croston_series_all.append(output.tail(1))



vec=pd.Series(croston_series_all).reset_index(drop=True)

vec=vec.astype('float64')
#Prepare final submission file

for i in range(int(sample_submission.shape[0]/2)):

   sample_submission.iloc[i,1:29]=vec[i]



sample_submission.to_csv('submission.csv',index=False)