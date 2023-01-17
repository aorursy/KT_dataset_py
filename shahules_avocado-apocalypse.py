import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar

from datetime import datetime

import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly import tools

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

avocado=pd.read_csv("../input/avocado.csv")
avocado=avocado.copy()
avocado.info()
print("shape of data",avocado.shape)
avocado.columns
avocado.isnull().sum()
avocado['year']=avocado['Date'].apply(lambda x : x.split("-")[0])
avocado['month']=avocado['Date'].apply(lambda x : calendar.month_name[datetime.strptime(x,"%Y-%m-%d").month])
avocado['day']=avocado['Date'].apply(lambda x : calendar.day_name[datetime.strptime(x,"%Y-%m-%d").weekday()])

avocado.head(7)
avocado[avocado['day']!='Sunday']
typeof=avocado.groupby('type')['Total Volume'].agg('sum')
values=[typeof['conventional'],typeof['organic']]
labels=['conventional','organic']

trace=go.Pie(labels=labels,values=values)
py.iplot([trace])
conv=avocado[avocado['type']=='conventional'].groupby('year')['AveragePrice'].agg('mean')
org=avocado[avocado['type']=='organic'].groupby('year')['AveragePrice'].agg('mean')

trace1=go.Bar(x=conv.index,y=conv,name="conventional",
             marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

trace2=go.Bar(x=conv.index,y=org,name="organic",
             marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

data=[trace1,trace2]
layout=go.Layout(barmode="group",title="Comaparing organic and conventional avocadro prices over years",
                yaxis=dict(title="mean price"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


conv=avocado[avocado['type']=='conventional'].groupby('year')['Total Volume'].agg('mean')
org=avocado[avocado['type']=='organic'].groupby('year')['Total Volume'].agg('mean')

trace1=go.Bar(x=conv.index,y=conv,name="conventional",
             marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

trace2=go.Bar(x=conv.index,y=org,name="organic",
             marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)



data=[trace1,trace2]

layout=go.Layout(barmode="group",title="Comaparing  mean Volume of organic and conventional avocadro  sold over years",
                yaxis=dict(title="Volume sold"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
trace3=go.Scatter(x=conv.index,y=conv,name="conventional")
trace4=go.Scatter(x=org.index,y=org,name='organic')
fig=tools.make_subplots(rows=1,cols=2)
fig.append_trace(trace3,1,1)
fig.append_trace(trace4,1,2)

fig['layout'].update(height=500, title="", barmode="stack", showlegend=True,yaxis=dict(title="Mean Volume sold"))
py.iplot(fig)
avocado['Date']=avocado['Date'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d').date())

date_16=avocado[avocado['year']=='2016'].sort_values(by='Total Volume')
date_17=avocado[avocado['year']=='2017'].sort_values(by='Total Volume')
trace1=go.Bar(x=date_16['Date'],y=date_16['Total Volume'],name="2016")
trace2=go.Bar(x=date_17['Date'],y=date_17['Total Volume'],name='2017')
data=[trace1,trace2]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data)
py.iplot(fig)
types=date_17.groupby('month')[['4046','4225','4770']].agg('sum')
types=types.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]
trace1=go.Bar(x=types.index,y=types['4046'],name=' PLU 4046')
trace2=go.Bar(x=types.index,y=types['4225'],name=' PLU 4225')
trace3=go.Bar(x=types.index,y=types['4770'],name=' PLU 4770')

data=[trace1,trace2,trace3]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data)
py.iplot(fig)


types1=date_16.groupby('month')[['4046','4225','4770']].agg('sum')
types1=types1.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]

trace1=go.Scatter(x=types.index,y=types['4046'],name=' 2017PLU 4046',line=dict(color='green'))
trace2=go.Scatter(x=types.index,y=types['4225'],name=' 2017PLU 4225',line=dict(color='green'))
trace3=go.Scatter(x=types1.index,y=types1['4046'],name=' 2016PLU 4046',mode='markers+lines',line=dict(color='blue'))
trace4=go.Scatter(x=types1.index,y=types1['4225'],name=' 2016PLU 4225',mode='markers+lines',line=dict(color='blue'))

data=[trace1,trace2,trace3,trace4]

fig=go.Figure(data=data)
py.iplot(fig)


price_16=date_16.groupby('month')['AveragePrice'].agg('mean')
price_16=price_16.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]

price_17=date_17.groupby('month')['AveragePrice'].agg('mean')
price_17=price_17.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]

price_15=avocado[avocado['year']=='2015'].groupby('month')['AveragePrice'].agg('mean')
price_15=price_15.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]


trace2=go.Scatter(x=price_17.index,y=price_17,name='2017')
trace1=go.Scatter(x=price_16.index,y=price_16,name='2016')
trace3=go.Scatter(x=price_15.index,y=price_15,name='2015')
data=[trace2,trace1,trace3]

fig=go.Figure(data=data)
py.iplot(fig)

price_15=avocado[avocado['year']=='2015']['AveragePrice']
price_16=avocado[avocado['year']=='2016']['AveragePrice']
price_17=avocado[avocado['year']=='2017']['AveragePrice']
price_18=avocado[avocado['year']=='2018']['AveragePrice']

trace1=go.Box(y=price_15,name="2015")
trace2=go.Box(y=price_16,name='2016')
trace3=go.Box(y=price_17,name='2017')
trace4=go.Box(y=price_18,name='2018')
data=[trace1,trace2,trace3,trace4]
layout=go.Layout(title='Box plot of price per avocado ',yaxis=dict(title='price in dollars'))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
price_15=avocado[avocado['year']=='2015']['Total Volume']
price_16=avocado[avocado['year']=='2016']['Total Volume']
price_17=avocado[avocado['year']=='2017']['Total Volume']
price_18=avocado[avocado['year']=='2018']['Total Volume']

trace1=go.Box(y=price_15,name="2015")
trace2=go.Box(y=price_16,name='2016')
trace3=go.Box(y=price_17,name='2017')
trace4=go.Box(y=price_18,name='2018')
data=[trace1,trace2,trace3,trace4]
layout=go.Layout(title=" sales in diiferent years ",yaxis=dict(title="Volume sold"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
avocado.groupby(['region','year'],as_index=False)['Total Volume'].agg('mean')

    
    
    










