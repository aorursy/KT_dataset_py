
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
#import chart_studio
import plotly.graph_objects as go
import plotly.graph_objs as go
# import cufflinks as cf
# #import plotly.offline
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# covid_df = pd.read_csv('/Users/manaswitamahan/Desktop/Covid/covid_19_data.csv')
time_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

pd.set_option('display.max_columns', None)  

time_df.tail()
time_df['Country/Region'].nunique()
df_temp= time_df.drop(['Province/State', 'Lat', 'Long'], axis = 1) 

for col in df_temp.columns[4:-1:]:
    df1= df_temp.groupby(['Country/Region'])[col].sum().reset_index(name='Sum')
print('Following new countries got added on 6th April as they had no case until 6th\n', df1[df1['Sum'] == 0])
df_country_count = []
for col in time_df.columns[4:]:
    #print('Column name:', col)
    countries = time_df[time_df[col] > 0]
    #print(countries)
    df1 = countries.groupby('Country/Region')[col].sum().reset_index(name='Sum')
    df_country_count.append({'Date': col, 'Country Count': df1['Sum'].count()})
    #print(df1['Sum'].count())
    #print(df1['Country/Region'].count())   

df_country_count= pd.DataFrame(df_country_count)
print(df_country_count.tail())

fig = go.Figure([go.Scatter(x=df_country_count['Date'],  y=df_country_count['Country Count'])])

fig.layout.update(title_text="Growth in Number of Countries with Confirmed Cases")

iplot(fig)# use plotly.offline.iplot for offline plot
# Edit the layout


death_df.head(20)
recovered_df.head()
df_daily_count = []
for col in death_df.columns[4:]:
    #print('Column name:', col)
    confirmed_cum = time_df[col].sum()
    death_cum = death_df[col].sum()
    recovered_cum = recovered_df[col].sum()
    df_daily_count.append({'Date' : col, 'Confirmed Cumulative': confirmed_cum, 'Death Cumulative': death_cum,
                          'Recovered Cumulative' :recovered_cum})
df_daily_count= pd.DataFrame(df_daily_count)

df_daily_count['Confirmed Daily']=  df_daily_count['Confirmed Cumulative'].diff() 
df_daily_count['Confirmed Percent Change']=  round((df_daily_count['Confirmed Daily'].pct_change() * 100),2)
df_daily_count['Death Daily']=  df_daily_count['Death Cumulative'].diff() 
df_daily_count['Death Percent Change']=  round(df_daily_count['Death Daily'].pct_change() * 100)
df_daily_count['Recovered Daily']=  df_daily_count['Recovered Cumulative'].diff() 
df_daily_count['Recovered Percent Change']=  round(df_daily_count['Recovered Daily'].pct_change() * 100)
fig = make_subplots(rows=3, cols=1)

fig.append_trace(go.Scatter(
    x=df_daily_count['Date'],
    y=df_daily_count['Confirmed Cumulative'],
    name ='Confirmed Daily',
    line=dict(color='firebrick', width=4)
), row=1, col=1)

fig.append_trace(go.Scatter(
    x=df_daily_count['Date'],
    y=df_daily_count['Death Cumulative'],
    name ='Death Daily',
    line=dict(color='green', width=4)
), row=2, col=1)

fig.append_trace(go.Scatter(
    x=df_daily_count['Date'],
    y=df_daily_count['Recovered Cumulative'],
    name = 'Recovered Daily',
    line=dict(color='royalblue', width=4)
), row=3, col=1)


fig.layout.update(height = 800, width = 900, title_text="Total Confirmed Cases Globally",
                    yaxis_title='Daily Cases')
iplot(fig)# use plotly.offline.iplot for offline plot


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_daily_count['Date'], y= df_daily_count['Confirmed Percent Change'], name='Confirmed Daily Percent',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=df_daily_count['Date'], y= df_daily_count['Death Percent Change'], name = 'Death Daily Change',
                         line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=df_daily_count['Date'], y=df_daily_count['Recovered Percent Change'], name='Recovered Daily Change',
                         line=dict(color='orange', width=4,
                              dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
))
# fig = go.Figure([go.Scatter(x=df_daily_count['Date'], y= df_daily_count['Confirmed Percent Change'], 
#                            name = 'Confirmed Daily Percent',  line=dict(color='royalblue', width=4))])


fig.layout.update(title_text="Global Daily Percent Change in Confirmed, Death Cases",
                    xaxis_title='Dates',
                    yaxis_title='Daily Percent Change')
iplot(fig)# use plotly.offline.iplot for offline plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_daily_count['Date'], y= df_daily_count['Confirmed Daily'], name='Confirmed Daily',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=df_daily_count['Date'], y= df_daily_count['Death Daily'], name = 'Death Daily',
                         line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=df_daily_count['Date'], y=df_daily_count['Recovered Daily'], name='Recovered Daily',
                         line=dict(color='orange', width=4,
                              dash='dash') # dash options include 'dash', 'dot', and 'dashdot'
))

fig.layout.update(title_text="Global Daily Confirmed, Recoved and Death Cases",
                    xaxis_title='Dates',
                    yaxis_title='Daily Cases')
iplot(fig)# use plotly.offline.iplot for offline plot
time_df = time_df.sort_values(by = 'Country/Region')
#select first and last columns only
time_df= time_df.iloc[:, [1] + [-1]]

#time_df['Country/Region'].unique()
#Find countries with maximum number of cases
top_countries = time_df.groupby(['Country/Region'], as_index= False).sum()
# top_countries= top_countries[['Country/Region','3/29/20']]
india = top_countries[top_countries['Country/Region'] == 'India']
india
#top_countries.head(20)
top_countries.columns[-1]
# top_countries= top_countries.sort_values(by = '3/29/20', ascending = False, inplace= True)
top_countries.sort_values(top_countries.columns[-1], ascending = False, inplace= True)


top_countries= top_countries.head(20)
top = pd.concat([india, top_countries])

top
population = pd.read_csv('../input/world-population/worldpopulation.csv')

 #pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

# Merge two Dataframes on single column 'ID'
result = top.merge(population, on='Country/Region')
#result = pd.merge(top_countries, population, how ='left', on=['Country/Region', 'Country/Region'])
result = pd.DataFrame(result)

# result.columns[1]
# result.iloc[:,1]


result['permillion cases'] = (result.iloc[:,1] / result['Population']) * 1000000
result.iloc[:,1]
result.rename(columns={ result.columns[1]: "last_day" }, inplace = True)
top= result.sort_values(by = 'last_day', ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=top['last_day'],
    y=top['Country/Region'],
    name='Total Cases',
    orientation='h',
     marker=dict(
         color='rgba(58, 71, 80, 0.6)',
         line=dict(color='rgba(58, 71, 80, 1.0)', width=10)
  )
))


fig.layout.update(barmode= 'stack',title_text="Top 20 Countries + India - Confirmed Cases",
                    xaxis_title='Dates',
                    yaxis_title='Top 20 Countries + India')


iplot(fig)
result = result.sort_values(by = 'permillion cases', ascending= False)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=result['permillion cases'],
    y=result['Country/Region'],
    name='Cases per one million',
    text = result['permillion cases'],
    orientation='h',
    hoverinfo="none",
    marker=dict(
        color='rgba(246, 78, 139, 0.6)'
        #line=dict(color='rgba(246, 78, 139, 1.0)', width=12)
     )
))
# fig.add_trace(go.Bar(
#     x=result.iloc[:,1],
#     y=result['Country/Region'],
#     name='Total Cases',
#     orientation='h',
# #     marker=dict(
# #         color='rgba(58, 71, 80, 0.6)',
# #         line=dict(color='rgba(58, 71, 80, 1.0)', width=10)
# #     )
# ))

fig.update_traces(texttemplate='%{text:.2s}', textposition= 'outside')

fig.layout.update(barmode= 'stack',title_text="Confirmed Cases per one million of Population in Top 20 Countries & India",
                  xaxis_title='Confirmed Covid-19 Cases',
                  yaxis_title='Top 20 Countries + India',
                  uniformtext_minsize=8, uniformtext_mode='hide')

# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')


iplot(fig, filename='/Users/manaswitamahan/Desktop/Covid/filename.png')

time_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
time_df = time_df.sort_values(by = 'Country/Region')
time_df = time_df.drop(['Lat', 'Long'], axis= 1)
countries = ['China', 'US', 'Italy', 'United Kingdom', 'Spain', 'Germany', 'France', 'India', 'Singapore', 'Korea, South']
selected = time_df[time_df['Country/Region'].isin(countries)]
confirmed_cases = selected.groupby('Country/Region').sum().reset_index()
confirmed_cases
confirmed_transposed = confirmed_cases.set_index('Country/Region').T.rename_axis('Date').reset_index()
confirmed_transposed.info()
# confirmed_transposed.plot()
# plt.show()


fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y= confirmed_transposed['China'], name='China',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y= confirmed_transposed['US'], name = 'US',
                         line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y=confirmed_transposed['Italy'], name='Italy',
                         line=dict(color='orange', width=4,dash='dash'))) # dash options include 'dash', 'dot', and 'dashdot'
fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y= confirmed_transposed['Germany'], name = 'Germany',
                         line=dict(color='green', width=4, dash ='dot')))
fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y= confirmed_transposed['Spain'], name = 'Spain',
                         line=dict(color='black', width=4, dash = 'dashdot')))
fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y=confirmed_transposed['France'], name='France',
                         line=dict(color='purple', width=4))) # dash options include 'dash', 'dot', and 'dashdot'
# fig.add_trace(go.Scatter(x=confirmed_transposed['Date'], y=confirmed_transposed['Korea, South'], name='S. Korea',
#                          line=dict(color='yellow', width=4)))


fig.layout.update(title_text="Confirmed growth day-on-day in Top 6",
                    xaxis_title='Dates',
                    yaxis_title='Top 6')
iplot(fig)# use plotly.offline.iplot for offline plot

#India take from 1st March to see the growth 
confirmed_transposed['Date'] = pd.to_datetime(confirmed_transposed['Date'])  
india_df = confirmed_transposed.loc[(confirmed_transposed['Date'] >= '3/11/20')]
india_df = india_df[['Date','India', 'Singapore', 'Korea, South']]
india_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=india_df['Date'], y= india_df['India'], name='India',
                         line=dict(color='firebrick', width=4)))
# fig.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Korea, South'], name='S. Korea',
#                          line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=india_df['Date'], y=india_df['Singapore'], name='Singapore',
                         line=dict(color='yellow', width=4)))



fig.layout.update(title_text="How the cases are growing in India vis-a-vis Singapore",
                    xaxis_title='Dates',
                    yaxis_title='India')

iplot(fig)
