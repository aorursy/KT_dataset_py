import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#datetime
from datetime import datetime

nifty_50 = pd.read_csv('../input/nifty-indices-dataset/NIFTY 50.csv',parse_dates=["Date"])
nifty_50.head()
nifty_50.info()
nifty_50.isnull().sum()
nifty_50.fillna(method='ffill',inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
         x=nifty_50['Date'],
         y=nifty_50['High'],
         name='High Price',
    line=dict(color='blue'),
    opacity=0.8))

fig.add_trace(go.Scatter(
         x=nifty_50['Date'],
         y=nifty_50['Low'],
         name='Low Price',
    line=dict(color='orange'),
    opacity=0.8))
        
    
fig.update_layout(title_text='NIFTY 50 High vs Close Trend',plot_bgcolor='rgb(250, 242, 242)',yaxis_title='Value')

fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(
         x=nifty_50['Date'],
         y=nifty_50['Close'],
         name='Closing Price',
    line=dict(color='blue'),
    opacity=0.8))

    
fig.update_layout(title_text='NIFTY 50 Closing Price',plot_bgcolor='rgb(250, 242, 242)',yaxis_title='Value')

fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(
         x=nifty_50['Date'],
         y=nifty_50['P/E'],
         name='P/E Ratio',
    line=dict(color='green'),
    opacity=0.8))

fig.add_trace(go.Scatter(
         x=nifty_50['Date'],
         y=nifty_50['P/B'],
         name='P/B Ratio',
    line=dict(color='orange'),
    opacity=0.8))
        
    
fig.update_layout(title_text='P/E vs P/B Ratio',plot_bgcolor='rgb(250, 242, 242)',yaxis_title='Value')

fig.show()
nifty_50_2019 = nifty_50[nifty_50['Date'] >= '2019-01-01']
nifty_50_2019.head()
df=nifty_50_2019
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.show()
fig = px.line(nifty_50_2019, x='Date', y='Close', title='Time Series with Range Slider and Selectors')

fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)',
    title='NIFTY_50 : Major single day falls -2019 onwards',
    yaxis_title='NIFTY 50 Stock',
    shapes = [dict(x0='2020-03-23', x1='2020-03-23', y0=0, y1=1, xref='x', yref='paper', line_width=2,opacity=0.3,line_color='red',editable=False),
             dict(x0='2019-09-3', x1='2019-09-3', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='red'),
             dict(x0='2020-02-1', x1='2020-02-1', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='red'),
             dict(x0='2020-03-12', x1='2020-03-12', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='red')],
    annotations=[dict(x='2020-03-23', y=0.5, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='Lockdown Phase-1 announced'),
                dict(x='2019-09-3', y=0.05, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='Multiple PSU Bank Merger Announcements'),
                dict(x='2020-02-1', y=0.5, xref='x', yref='paper',
                    showarrow=False, xanchor='right', text='Union Budget,coronavirus pandemic'),
                dict(x='2020-03-12', y=0.3, xref='x', yref='paper',
                    showarrow=False, xanchor='right', text='Coronavirus declared Pandemic by WHO')]
)
fig.show()
fig = px.line(nifty_50_2019, x='Date', y='Close', title='Time Series with Range Slider and Selectors')

fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)',
    title='NIFTY_50 : Major single day gains -2019 onwards',
    yaxis_title='NIFTY 50 Stock',
    shapes = [dict(x0='2019-05-20', x1='2019-05-20', y0=0, y1=1, xref='x', yref='paper', line_width=2,opacity=0.3,line_color='green',editable=False),
             dict(x0='2020-05-23', x1='2020-05-23', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green'),
             dict(x0='2019-09-20', x1='2019-09-20', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green'),
             dict(x0='2020-04-07', x1='2020-04-07', y0=0, y1=1, xref='x', yref='paper',line_width=3,opacity=0.3,line_color='green')],
    annotations=[dict(x='2019-05-20', y=0.54, xref='x', yref='paper',
                    showarrow=False, xanchor='right', text='Exit-Polls predict majority'),
                 dict(x='2019-05-20', y=0.5, xref='x', yref='paper',
                    showarrow=False, xanchor='right', text='for BJP government'),
                dict(x='2019-09-3', y=0.08, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='2019 General Elections'),
                 dict(x='2019-09-3', y=0.05, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='results announced'),
                dict(x='2019-09-20', y=0.54, xref='x', yref='paper',
                    showarrow=False, xanchor='left', text='cut in the corporate tax rate announced'),
                dict(x='2020-04-07', y=0.3, xref='x', yref='paper',
                    showarrow=False, xanchor='right', text='Italy Coronavirus Nos went down')]
)
fig.show()
nifty_auto = pd.read_csv('../input/nifty-indices-dataset/NIFTY AUTO.csv',parse_dates=["Date"])
nifty_bank = pd.read_csv('../input/nifty-indices-dataset/NIFTY BANK.csv',parse_dates=["Date"])
nifty_fmcg = pd.read_csv('../input/nifty-indices-dataset/NIFTY FMCG.csv',parse_dates=["Date"])
nifty_IT = pd.read_csv('../input/nifty-indices-dataset/NIFTY IT.csv',parse_dates=["Date"])
nifty_metal = pd.read_csv('../input/nifty-indices-dataset/NIFTY METAL.csv',parse_dates=["Date"])
nifty_pharma = pd.read_csv('../input/nifty-indices-dataset/NIFTY PHARMA.csv',parse_dates=["Date"])


#Fill in missing values
nifty_auto.fillna(method='ffill',inplace=True)
nifty_bank.fillna(method='ffill',inplace=True)
nifty_fmcg.fillna(method='ffill',inplace=True)
nifty_IT.fillna(method='ffill',inplace=True)
nifty_metal.fillna(method='ffill',inplace=True)
nifty_pharma.fillna(method='ffill',inplace=True)

nifty_auto_2019 = nifty_auto[nifty_auto['Date'] > '2019-12-31']
nifty_bank_2019 = nifty_bank[nifty_bank['Date'] > '2019-12-31']
nifty_fmcg_2019 = nifty_fmcg[nifty_fmcg['Date'] > '2019-12-31']
nifty_IT_2019 = nifty_IT[nifty_IT['Date'] > '2019-12-31']
nifty_metal_2019 = nifty_metal[nifty_metal['Date'] > '2019-12-31']
nifty_pharma_2019 = nifty_pharma[nifty_pharma['Date'] > '2019-12-31']

d = {'NIFTY Auto index': nifty_auto_2019['Close'].values, 
     'NIFTY Bank index': nifty_bank_2019['Close'].values,
     'NIFTY FMCG index': nifty_fmcg_2019['Close'].values,
     'NIFTY IT index': nifty_IT_2019['Close'].values,
     'NIFTY Metal index': nifty_metal_2019['Close'].values,
     'NIFTY Pharma index': nifty_pharma_2019['Close'].values,
    }



df = pd.DataFrame(data=d)
df.index=nifty_auto_2019['Date']
df.head()
df.iplot(kind='box')
fig = df.iplot(asFigure=True, subplots=True, subplot_titles=True, legend=False)
fig.show()
fig = df.iplot(asFigure=True, hline=[2,4], vline=['2020-03-23'])
fig.show()
fig = df.iplot(asFigure=True,
               vspan={'x0':'2020-03-23','x1':'2020-04-14',
                      'color':'rgba(30,30,30,0.3)','color':'teal','fill':True,'opacity':.4})

fig.show()
fig = df.iplot(asFigure=True,
               vspan={'x0':'2020-04-15','x1':'2020-05-03',
                      'color':'rgba(30,30,30,0.3)','color':'red','fill':True,'opacity':.4})

fig.show()
df_a=df['NIFTY Pharma index']
max_val=df_a.max()
min_val=df_a.min()
max_date=df_a[df_a==max_val].index[0].strftime('%Y-%m-%d')
min_date=df_a[df_a==min_val].index[0].strftime('%Y-%m-%d')
shape1=dict(kind='line',x0=max_date,y0=max_val,x1=min_date,y1=min_val,color='blue',width=2)
shape2=dict(kind='rect',x0=max_date,x1=min_date,fill=True,color='gray',opacity=.3)

df_a.iplot(shapes=[shape1,shape2])
