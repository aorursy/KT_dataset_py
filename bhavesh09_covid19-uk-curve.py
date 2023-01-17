import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df1= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df = pd.melt(df, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
       var_name='Date',
       value_name = 'cum_deaths')
df['Date'] = pd.to_datetime( df['Date'])
df = df.rename(columns={'Province/State' : 'State',
                  'Country/Region' : 'Country'})
df = df.groupby(['Country','Date'])['cum_deaths'].sum().reset_index()
df1 = pd.melt(df1, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
       var_name='Date',
       value_name = 'total_cases')
df1['Date'] = pd.to_datetime( df1['Date'])
df1 = df1.rename(columns={'Province/State' : 'State',
                  'Country/Region' : 'Country'})
df1 = df1.groupby(['Country','Date'])['total_cases'].sum().reset_index()
df['Prev_day_deaths'] = df.groupby('Country')['cum_deaths'].shift(1)
df['Prev_day_deaths'] = df['Prev_day_deaths'].fillna(0)
df['New_deaths'] = df['cum_deaths'] - df['Prev_day_deaths']
df['day_since_10_d'] = df[df.cum_deaths>=10].groupby('Country')['Date'].rank()
df['day_since_100_d'] = df[df.cum_deaths>=100].groupby('Country')['Date'].rank()
df['dp_deaths'] = df['New_deaths']/df['Country'].map(df.groupby('Country')['cum_deaths'].max())
df['rolling_mean_dp'] = df['dp_deaths'].rolling(window=5).mean()
df['rolling_mean'] = df['New_deaths'].rolling(window=5).mean()
df['rolling_mean_7'] = df['New_deaths'].rolling(window=7).mean()
df['rolling_new_cases'] = df['New_deaths'].rolling(window=5).mean()
trace1 = go.Bar(x=df[df.Country=='United Kingdom'].Date,
                y=df[df.Country=='United Kingdom'].New_deaths,
                name="UK Daily Deaths",
               text="Date")
trace2 = go.Scatter(x=df[df.Country=='United Kingdom'].Date,
                y=df[df.Country=='United Kingdom'].rolling_mean,
                mode="lines+markers",   
                name="UK Death Distribution",
               text="Date")
layout = {'template' : "plotly_dark",
         "legend.orientation" : "h"}
fig = dict(data=[trace1,trace2], layout=layout)
iplot(fig)
poi_countries = df[df.day_since_10_d>=1].Country.unique() 
trace1 = go.Bar()
poi_countries = df[df.day_since_10_d>=1].Country.unique() # Only considerign countries which crossed death toll 10 before 7 days at least
fig = px.line(df[df.Country.isin(poi_countries)], 
              x="day_since_10_d", y="cum_deaths", color='Country')
'''
fig = px.scatter(df[(df['day_since_10_d'] == df['Country'].map(df.groupby(['Country'])['day_since_10_d'].max())) &
                    (df.Country.isin(poi_countries))],
                x="day_since_10_d", y="cum_deaths", text='Country')
'''
fig.add_trace(go.Scatter(x=[1, 21], y=[10, 10000], name='Double by 7 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[1, 43], y=[10, 10000], name='Double by 14 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[1, 64], y=[10, 10000], name='Double by 21 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.update_layout(title='VIRUS PANDEMIC',
                   xaxis_title='Number of Days Since 10th Death',
                   yaxis_title='No. of Deaths',
                   yaxis_type="log",
                   template = "plotly_dark")
iplot(fig)
poi_countries = df[df.day_since_100_d>=1].Country.unique() # Only considerign countries which crossed death toll 10 before 7 days at least
fig = px.line(df[df.Country.isin(poi_countries)], 
              x="day_since_100_d", y="cum_deaths", color='Country')
fig.update_layout(title='VIRUS PANDEMIC',
                   xaxis_title='Number of Days Since 100th Death',
                   yaxis_title='No. of Deaths',
                   yaxis_type="log",
                   template = "plotly_dark")
fig.add_trace(go.Scatter(x=[1, 40], y=[100, 819200], name='Double by 3 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[1, 40], y=[100, 86108], name='Double by 4 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))
fig.add_trace(go.Scatter(x=[1, 40], y=[100, 2286], name='Double by 5 days', line=dict(dash='dash', color=('rgb(200, 200, 200)'))))

iplot(fig)
trace1 = go.Bar(x=df[df.Country=='India'].Date,
                y=df[df.Country=='India'].New_deaths,
                name="UK Daily Deaths",
               text="Date")
trace2 = go.Scatter(x=df[df.Country=='India'].Date,
                y=df[df.Country=='India'].rolling_mean,
                mode="lines+markers",   
                name="India Death Trend",
               text="Date")
layout = {'template' : "plotly_dark",
         "legend.orientation" : "h"}
fig = dict(data=[trace1,trace2], layout=layout)
iplot(fig)
trace1 = go.Bar(x=df[df.Country=='United Kingdom'].Date,
                y=df[df.Country=='United Kingdom'].New_deaths,
                name="UK Daily Deaths",
               text="Date")
layout = {'template' : "plotly_dark",
         "legend.orientation" : "h"}
fig = dict(data=[trace1], layout=layout)
iplot(fig)
trace1 = go.Bar(x=df[df.Country=='United Kingdom'].Date,
                y=df[df.Country=='United Kingdom'].New_deaths,
                yaxis='y1',
                name="UK Daily Deaths",
               text="Date")
trace2 = go.Scatter(x=df[df.Country=='United Kingdom'].Date,
                y=df[df.Country=='United Kingdom'].rolling_mean,
                yaxis='y2',
                mode="lines+markers",   
                name="UK Death Distribution",
               text="Date")
layout = {'template' : "plotly_dark",
         "legend.orientation" : "h",
         "yaxis2" : {"overlaying" : "y",
                    "side" : "right"}}
fig = dict(data=[trace1,trace2], layout=layout)
iplot(fig)
trace1 = go.Bar(x=df[df.Country=='Italy'].Date,
                y=df[df.Country=='Italy'].New_deaths,
                yaxis='y1',
                name="Italy Daily Deaths",
               text="Date")
trace2 = go.Scatter(x=df[df.Country=='Italy'].Date,
                y=df[df.Country=='Italy'].rolling_mean,
                yaxis='y2',
                mode="lines+markers",   
                name="Italy Death Distribution (5 SMA)",
               text="Date")
layout = {'template' : "plotly_dark",
         "legend.orientation" : "h",
         "yaxis2" : {"overlaying" : "y",
                    "side" : "right"}}
fig = dict(data=[trace1,trace2], layout=layout)
iplot(fig)
poi_countries = df[df.day_since_100_d>=1].Country.unique() # Only considerign countries which crossed death toll 10 before 7 days at least
fig = px.line(df[(df.day_since_100_d.between(1,60)) & (df.Country.isin(poi_countries))], 
              x="day_since_100_d", y="rolling_mean_7", color='Country')
fig.update_layout(title='VIRUS PANDEMIC',
                   xaxis_title='Number of Days Since 100th Death',
                   yaxis_title='No. of Deaths',
                   template = "plotly_dark")
iplot(fig)
df['rolling_mean_dp'] = df['rolling_mean_dp'].fillna(0)
df.head()
poi_countries = df[df.day_since_100_d>=1].Country.unique() # Only considerign countries which crossed death toll 10 before 7 days at least
fig = px.line(df[(df.Country.isin(poi_countries))], 
              x="Date", y="rolling_mean", color='Country')
for c in poi_countries:
    fig.add_trace(go.Scatter(x=df[(df.Country==c) & (df.Date=='2020-04-14')].Date,
                             y=df[(df.Country==c) & (df.Date=='2020-04-14')].rolling_mean,
                            mode="text",
                            text=c))

fig.update_layout(title='VIRUS PANDEMIC',
                   xaxis_title='Date',
                   yaxis_title='No. of Deaths Rolling Mean(5 SMA)',
                   template = "plotly_dark")
iplot(fig)
poi_countries = df[df.day_since_100_d>=1].Country.unique() # Only considerign countries which crossed death toll 10 before 7 days at least
fig = go.Figure()
for c in poi_countries:
    fig.add_trace(go.Scatter(x=df[df.Country==c].Date,
                             y=df[df.Country==c].rolling_mean,
                            mode="lines"))
    fig.add_trace(go.Scatter(x=df[(df.Country==c) & (df.Date=='2020-04-14')].Date,
                             y=df[(df.Country==c) & (df.Date=='2020-04-14')].rolling_mean,
                            mode="text",
                            text=c))
fig.update_layout(
    showlegend=False,
    template = "plotly_dark")
fig.show()
df[df.Date=='2020-04-23'].sort_values(by='New_deaths', ascending=False)[:20]
