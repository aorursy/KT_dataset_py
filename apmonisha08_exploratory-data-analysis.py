# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Importing plotly libraries
import plotly
import plotly.offline as pltoff
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')

df=pd.read_csv("../input/train.csv",date_parser=dateparse)
df.head(2)
# Box plot
s=list(np.array(np.repeat(0,11)))
box_plot=[]
for i in range(1,len(s)):
    s[i]=df[df.store == i].sales

trace0 = go.Box(y=s[1],name = 'store1',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace1 = go.Box(y=s[2],name = 'store2',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace2 = go.Box(y=s[3],name = 'store3',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace3 = go.Box(y=s[4],name = 'store4',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace4 = go.Box(y=s[5],name = 'store5',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace5 = go.Box(y=s[6],name = 'store6',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace6 = go.Box(y=s[7],name = 'store7',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace7 = go.Box(y=s[8],name = 'store8',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace8 = go.Box(y=s[9],name = 'store9',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)
trace9 = go.Box(y=s[10],name = 'store10',marker = dict(color = 'rgba(93, 164, 214, 0.5)',),boxmean=True)

data = [trace0, trace1,trace2, trace3,trace4, trace5,trace6, trace7,trace8, trace9]

layout = go.Layout(autosize= True,title= 'Distribution of number of  per store',xaxis= dict(title= 'Stores',),
                      yaxis=dict(title= 'Number of Sales',))

fig = go.Figure(data=data, layout=layout)
pltoff.iplot(fig)
# pivot table 
sales_pivoted_df=pd.pivot_table(df, values='sales', index=['store'], columns=['item'], aggfunc=np.sum)
sales_pivoted_df
# pivot table
item_sales=pd.pivot_table(df, values='sales', index=['item'], aggfunc=np.sum)
item_sales=item_sales.sort_values('sales', ascending=False).reset_index()
print('Most sold items:\n',item_sales.head(3))
print('Least sold items:\n',item_sales.tail(2))
# Bar plot of sales per item

item_sales_data = [go.Bar(
    x=[i for i in range(1, 51)],
    y=item_sales.sales.values,
    marker=dict(
        color='rgba(93, 164, 214, 0.5)'),
    text = item_sales.item.values
)]

item_sales_layout = go.Layout(
    autosize= True,
    title= 'Sales per item',
    xaxis= dict(title= 'Items',ticklen= 10,),
    yaxis=dict(title= 'Number of Sales',ticklen= 10,),
)

fig = go.Figure(data=item_sales_data,layout=item_sales_layout)
pltoff.iplot(fig)
# Scatt
store_item=df[ (df.store == 1)  & (df.item == 5) ]

store_item_fig = [go.Scatter(x=store_item.date,y=store_item.sales)]
lay = go.Layout(title = 'Sales trend of Store 1 and Item 5',xaxis= dict(title= 'Time'), yaxis=dict(title= 'Number of Sales'),)
fig = go.Figure(data=store_item_fig, layout=lay)
pltoff.iplot(fig)
# Scatter plot
store_id = range(1,11) 
item_id = np.repeat(1,10)    
store_same_item=df[ (df.store.isin(store_id))  & (df.item.isin(item_id)) ]

store_same_item_plot = []
for s,i in zip(store_id, item_id):
    d = store_same_item[store_same_item.store == s]
    store_same_item_plot.append(go.Scatter(x=d.date, y=d.sales, name = "Store:" + str(s) + ",Item:" + str(i)))

lay = go.Layout(title = 'Sales trend of Item 1 at different stores')
fig = go.Figure(data=store_same_item_plot, layout=lay)
pltoff.iplot(fig)
# Finding the day of week with the dates available
dat=df.copy()
dat['date'] = pd.to_datetime(dat['date'])
dat['day_of_week'] = dat['date'].dt.weekday_name

sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
sorterIndex = dict(zip(sorter,range(len(sorter))))
dat['day_id'] = dat['day_of_week'].map(sorterIndex)

dat=pd.DataFrame(dat.sort_values('day_id', ascending=True))
dat.head(5)
# pivot table
dat_pivot=pd.pivot_table(dat, values='sales', index=['day_id'], columns=['store'], aggfunc=np.sum)
dat_pivot['total_sales_day_of_week'] = dat_pivot.sum(axis=1)
dat_pivot.index=sorter
dat_pivot
# Bar plot 
data = [go.Bar(x=sorter,y=dat_pivot['total_sales_day_of_week'],marker = dict(color = 'rgba(93, 164, 214, 0.5)',))]
layout = go.Layout(title='total_sales_day_of_week',xaxis= dict(title= 'Day of week',),yaxis=dict(title= 'Number of Sales',),)
fig = go.Figure(data=data,layout=layout)
pltoff.iplot(fig)
# Subsetting sundays
sunday_store=dat[(dat.day_of_week == 'Sunday')]
sunday_store['date'] =pd.to_datetime(sunday_store.date)
sunday_store=sunday_store.sort_values(by='date')
sunday_store.head(3)
# Scatter plot
sunday_store_data = [go.Scatter(x=sunday_store.date,y=sunday_store.sales)]
lay = go.Layout(title = 'Sales trend of Sunday',xaxis= dict(title='Time Period',),yaxis=dict(title='Number of Sales',),)
fig = go.Figure(data=sunday_store_data, layout=lay)
pltoff.iplot(fig)
#Subsetting sundays
monday_store=dat[(dat.day_of_week == 'Monday')]
monday_store['date'] =pd.to_datetime(monday_store.date)
monday_store=monday_store.sort_values(by='date')
monday_store.head(3)
# Scatter plot
monday_store_data = [go.Scatter(x=monday_store.date,y=monday_store.sales)]
lay = go.Layout(title = 'Sales trend of Monday',xaxis= dict(title='Time Period',),yaxis=dict(title='Number of Sales',),)
fig = go.Figure(data=monday_store_data, layout=lay)
pltoff.iplot(fig)
month=dat.copy()
month=month.sort_values(by='date')

item_pivot=pd.pivot_table(month, values='sales', index=['date'], columns=['item'], aggfunc=np.sum)

# Resampling monthly 'M'
month_pivot=item_pivot.resample('M', how=[np.sum]).reset_index()
month_pivot.columns = month_pivot.columns.get_level_values(0)
month_pivot.head(5)
# Graphical representation
month_item1 = [go.Scatter(x=month_pivot.date,y=month_pivot[1])]

lay = go.Layout(title = 'Sales trend of Item 1 monthly over the period',xaxis= dict(title='Time Period',),yaxis=dict(title='Number of Sales',),)
fig = go.Figure(data=month_item1, layout=lay)
pltoff.iplot(fig)