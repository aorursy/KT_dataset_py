import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as exp
import plotly.graph_objects as go

import pandas as pd
import numpy as np

calendar = "/kaggle/input/m5-forecasting-accuracy/calendar.csv"
sales_train_validation = "/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv"
sell_prices = "/kaggle/input/m5-forecasting-accuracy/sell_prices.csv"


calendar_ = pd.read_csv(calendar, delimiter=",")
sales_train_validation_ = pd.read_csv(sales_train_validation, delimiter=",")
sell_prices_ = pd.read_csv(sell_prices, delimiter=",")
print(calendar_.shape)
calendar_.head()
print(sales_train_validation_.shape)
sales_train_validation_.head()
print(sell_prices_.shape)
sell_prices_.head()
groups = sales_train_validation_.groupby(['cat_id'])
counts_dict = {}
for name, group in groups:
    counts_dict[name] = len(group)
df = pd.DataFrame(counts_dict.items(), columns=['category', 'value'])
fig = exp.pie(df, values='value', names='category', title='Category wise sales of items')
fig.show()

groups = sales_train_validation_.groupby(['state_id'])
counts_dict = {}
for name, group in groups:
    counts_dict[name] = len(group)
df = pd.DataFrame(counts_dict.items(), columns=['category', 'value'])
fig = exp.pie(df, values='value', names='category', title='State wise sales of items')
fig.show()
date_dict = pd.Series(calendar_.date.values,index=calendar_.d).to_dict()
dates = list(date_dict.values())[0:1913] # we have sales data for 1913 days 

def plot_time_series(row):
    
    daily_sales = row.iloc[6:].values
    
    
    df = pd.DataFrame(
    {'daily sales': daily_sales,
     'date': dates,
    })

    fig = exp.line(df,x="date",y="daily sales")
    
    fig.update_layout(
    title={
        'text': "Daily Unit Sales of a particular item from 2011 - 2016",
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    
    fig.show()

plot_time_series(sales_train_validation_.iloc[1,:])
# plot a random sample of 10 points

df_sample = sales_train_validation_.sample(n=10, replace=False)
df_sample_ = pd.DataFrame(columns = ['item id','daily sales','date'])

colors = (exp.colors.sequential.Plasma)

def plot_sample(df_sample):
    
    fig = go.Figure()
    count = 0
    
    for index,row in df_sample.iterrows():
        
        
        item_id = row[1]
        daily_sales = row.iloc[6:].values
           
        fig.add_trace(go.Scatter(x=dates, y=daily_sales,
                    mode='lines+markers',
                    name=item_id, marker = dict(color=colors[count])))
        
        """ uncomment to plot a sample of more than 10 points
        fig.add_trace(go.Scatter(x=dates, y=daily_sales,
               mode='lines+markers',
               name=item_id))"""
        
        count+=1
    
    fig.update_layout(
    autosize=False,
    width=1200,
    height=500, title={
        'text': "Daily Unit Sales of items from 2011 - 2016",
        'y':1,
        'x':0.3,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    fig.show()

plot_sample(df_sample)        
        
        
### Moving Average

def plot_moving_avg_series(row):
    
    item_id = row.iloc[1]
    
    daily_sales_ = row.iloc[6:].values
    daily_sales = list(map(int, daily_sales_))
    
    df = pd.DataFrame(
    {'daily sales': daily_sales,
     'date': dates,
    })
    
    rolling = df.rolling(window=30)
    moving_avg = rolling.mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=dates, y=daily_sales,mode='lines+markers',name=item_id))
    
    fig.add_trace(go.Scatter(x=dates, y=moving_avg['daily sales'],mode='lines+markers',name=item_id))

    
    fig.update_layout(
    autosize=False,
    width=1500,
    height=500,
    title={
        'text': "Daily Unit Sales of a particular item from 2011 - 2016",
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    
    fig.show()
    
plot_moving_avg_series(sales_train_validation_.iloc[1,:])
def plot_moving_avg_sample(df_sample):
    
    fig = go.Figure()
    
    for index,row in df_sample.iterrows():
        
        
        item_id = row[1]
        
        daily_sales_ = row.iloc[6:].values
        daily_sales = list(map(int, daily_sales_))
    
        df = pd.DataFrame(
            {'daily sales': daily_sales,
             'date': dates,
            })
    
        rolling = df.rolling(window=30)
        moving_avg = rolling.mean()

        
        fig.add_trace(go.Scatter(x=dates, y=moving_avg['daily sales'],
               mode='lines+markers',
               name=item_id))
        
    fig.update_layout(
    autosize=False,
    width=1200,
    height=500,
    title={
        'text': "Daily Unit Sales of a sample from 2011 - 2016",
        'y':1,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="No. of Items Sold",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
    
    fig.show()

plot_moving_avg_sample(df_sample)
