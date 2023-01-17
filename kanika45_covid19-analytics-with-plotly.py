import pandas as pd 
import seaborn as sns
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import warnings
warnings.simplefilter("ignore")
df=pd.read_csv('../input/covid_19_data.csv')

df.columns=['SNo', 'Date', 'Province/State', 'Country/Region',
       'Last Update', 'Confirmed', 'Deaths', 'Recovered']
df['Province/State']=df['Province/State'].fillna('')
df['Date']=pd.to_datetime(df['Date'])
del df['Last Update']
del df['Province/State']
del df['SNo']
df['Active']=df['Confirmed']-df['Deaths']-df['Recovered']
def make_4_plots(table):
    fig = make_subplots(rows=1, cols=4, subplot_titles=("Confirmed", "Deaths", "Recovered",'Active'))
    trace1 = go.Scatter(
                    x=table['Date'],
                    y=table['Confirmed'],
                    name="Confirmed",
                    line_color='orange',
                    mode='lines+markers',
                    opacity=0.8)
    trace2 = go.Scatter(
                    x=table['Date'],
                    y=table['Deaths'],
                    name="Deaths",
                    line_color='red',
                    mode='lines+markers',
                    opacity=0.8)

    trace3 = go.Scatter(
                    x=table['Date'],
                    y=table['Recovered'],
                    name="Recovered",
                    mode='lines+markers',
                    line_color='green',
                    opacity=0.8)

    trace4 = go.Scatter(
                    x=table['Date'],
                    y=table['Active'],
                    name="Active",
                    line_color='blue',
                    mode='lines+markers',
                    opacity=0.8)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 3)
    fig.append_trace(trace4, 1, 4)
    fig.update_layout(title_text = 'CoronaVirus Spead in '+str(country_name)   )
    fig.show()
def plot_corona_spread(table):
    fig = make_subplots(rows=1, cols=1)
    trace1 = go.Scatter(
                    x=table['Date'],
                    y=table['Confirmed'],
                    name="Confirmed",
                    line_color='orange',
                    mode='lines+markers',
                    opacity=0.8)
    trace2=go.Bar( x=table['Date'],
                    y=table['Deaths'],
                    name="Deaths")

    trace3 = go.Scatter(
                    x=table['Date'],
                    y=table['Recovered'],
                    name="Recovered",
                    mode='lines',
                    line_color='green',
                    opacity=1)

    trace4 = go.Scatter(
                    x=table['Date'],
                    y=table['Active'],
                    name="Active",
                    line_color='blue',
                    mode='markers',
                    opacity=0.8)
    fig.append_trace(trace1, 1, 1)
    fig.add_trace(trace2, 1, 1)
    fig.add_trace(trace3, 1, 1)
    fig.add_trace(trace4, 1, 1)
    fig.update_layout(title_text = 'CoronaVirus Spead in '+ str(country_name),template="plotly_dark",font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white')   )
    fig.show()
    
def daily_cases(table):
    arr=[]
    for i in range(0,table.shape[0]):
        if i==0:
            arr.append(0)
        else:
            arr.append(table.iloc[i]['Confirmed']-table.iloc[i-1]['Confirmed'])
    table['daily_Confirmed']=arr
    arr=[]
    for i in range(0,table.shape[0]):
        if i==0:
            arr.append(0)
        else:
            arr.append(table.iloc[i]['Deaths']-table.iloc[i-1]['Deaths'])
    table['daily_Deaths']=arr
    arr=[]
    for i in range(0,table.shape[0]):
        if i==0:
            arr.append(0)
        else:
            arr.append(table.iloc[i]['Recovered']-table.iloc[i-1]['Recovered'])
    table['daily_Recoveries']=arr
    arr=[]
    for i in range(0,table.shape[0]):
        if i==0:
            arr.append(0)
        else:
            arr.append(table.iloc[i]['Active']-table.iloc[i-1]['Active'])
    table['daily_Active']=arr
    fig = make_subplots(rows=1, cols=1)
    trace1 = go.Scatter(
                    x=table['Date'],
                    y=table['daily_Confirmed'],
                    name="Confirmed",
                    line_color='orange',
                    mode='lines+markers',
                    opacity=0.8)
    trace2=go.Bar( x=table['Date'],
                    y=table['daily_Deaths'],
                    name="Deaths")

    trace3 = go.Scatter(
                    x=table['Date'],
                    y=table['daily_Recoveries'],
                    name="Recovered",
                    mode='lines',
                    line_color='green',
                    opacity=1)

    trace4 = go.Scatter(
                    x=table['Date'],
                    y=table['daily_Active'],
                    name="Active",
                    line_color='blue',
                    mode='lines',
                    opacity=0.8)
    fig.append_trace(trace1, 1, 1)
    fig.add_trace(trace2, 1, 1)
    fig.add_trace(trace3, 1, 1)
    fig.add_trace(trace4, 1, 1)
    fig.update_layout(title_text = 'CoronaVirus Spead Per Day in '+str(country_name),template="plotly_dark",font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white')   )
    fig.show()
def doubling_time(table):
    table['TenthDate']=pd.to_datetime(table[table['Deaths']>9]['Date'].min())
    table['Diff']=(table['Date'] -table['TenthDate']).dt.days.astype('int64')
    table['DoublingTime']=table['Deaths']/table['Diff']

    arr=[]
    for i in range(0,table.shape[0]):
        if i==0:
            arr.append(0)
        else:
            arr.append(70/(((table.iloc[i]['Deaths']-table.iloc[i-1]['Deaths'])/table.iloc[i-1]['Deaths'])* 100))
    table['GrowthRate']=arr
    # table['GrowthRate']=
    table["Mortality Rate"]=(table["Deaths"]/table["Confirmed"])*100
    table["Recovery Rate"]=(table["Recovered"]/table["Confirmed"])*100
    del table['Diff']
    del table['TenthDate']
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Doubling Time (in log)", "Mortality vs Recovery Rate"))
    trace1 = go.Scatter(
                    x=table['Date'],
                    y=table['Recovery Rate'],
                    name="Recovery Rate",
                    line_color='green',
                    mode='lines',
                    opacity=0.8)
    trace2 = go.Scatter(
                    x=table['Date'],
                    y=table['Mortality Rate'],
                    name="Mortality Rate",
                    mode='lines',
                    line_color='red',
                    opacity=1)

    trace3 = go.Scatter(
                    x=table['Date'],
                    y=table['GrowthRate'],
                    name="Death Growth Rate",
                    line_color='pink',
                    mode='lines',
                    opacity=0.8)
    trace4 = go.Scatter(
                    x=table['Date'],
                    y=table['DoublingTime'],
                    name="Doubling Time",
                    line_color='blue',
                    mode='lines',
                    opacity=0.8)
    fig.append_trace(trace4,1,1)
    fig.append_trace(trace1,1,2)
    fig.add_trace(trace2,1,2)
    fig.update_yaxes(row=1,col=1,type="log")
    fig.update_layout(  title_text = 'Infection Trajectory in '+str(country_name),template="plotly_dark",font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white')   )
    fig.show()
    
datewise=df.groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()
country_name="World"
plot_corona_spread(datewise)
top10=df[df["Date"]==df["Date"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)
sns.set_style("dark")
sns.set_context('poster')
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(27,20))
top_15_confirmed=top10.sort_values(["Confirmed"],ascending=False).head(15)
top_15_deaths=top10.sort_values(["Deaths"],ascending=False).head(15)
sns.barplot(x=top_15_confirmed["Confirmed"],y=top_15_confirmed.index,ax=ax1,palette="inferno")
ax1.set_title("Top 15 countries as per Number of Confirmed Cases")
sns.barplot(x=top_15_deaths["Deaths"],y=top_15_deaths.index,ax=ax2,palette="inferno")
ax2.set_title("Top 15 countries as per Number of Death Cases")
china_datewise=df[df['Country/Region']=='Mainland China'].groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()
country_name="China"

plot_corona_spread(china_datewise)
italy_datewise=df[(df['Country/Region']=='Italy') & (df['Confirmed']>0.0)].groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()
country_name="Italy"

plot_corona_spread(italy_datewise)
daily_cases(italy_datewise)
usa_datewise=df[(df['Country/Region']=='US') & (df['Confirmed']> 0.0)].groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()
country_name="USA"
plot_corona_spread(usa_datewise)
daily_cases(usa_datewise)
doubling_time(usa_datewise)
spain_datewise=df[(df['Country/Region']=='Spain')&(df['Confirmed']>0.0)].groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()
country_name="spain"
plot_corona_spread(spain_datewise)
daily_cases(spain_datewise)
df1=df[df['Confirmed']>100.0].groupby(['Date','Country/Region'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()

import plotly.express as px
fig = px.scatter(df1, x='Date', y='Confirmed', color='Country/Region',template="plotly_dark")
fig.show()
rest_of_world=df[df['Country/Region'].isin(['Mainland China','US','Italy','Spain','UK','South Korea'])].groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()

sk_datewise=df[(df['Country/Region']=='South Korea') &(df['Confirmed']>0.0)].groupby(['Date'])['Confirmed','Deaths', 'Recovered', 'Active'].sum().reset_index()

fig = make_subplots(rows=1, cols=1)

trace1 = go.Scatter(
                x=china_datewise['Date'],
                y=china_datewise['Confirmed'],
                name="China",
                line_color='orange',
                mode='lines+markers',
                opacity=0.8)
trace3 = go.Scatter(
                x=usa_datewise['Date'],
                y=usa_datewise['Confirmed'],
                name="USA",
                mode='lines',
                line_color='red',
                opacity=1)

trace4 = go.Scatter(
                x=italy_datewise['Date'],
                y=italy_datewise['Confirmed'],
                name="Italy",
                line_color='blue',
                mode='lines',
                opacity=0.8)
trace5 = go.Scatter(
                x=spain_datewise['Date'],
                y=spain_datewise['Confirmed'],
                name="Spain",
                line_color='green',
                mode='lines',
                opacity=0.8)
trace7 = go.Scatter(
                x=sk_datewise['Date'],
                y=sk_datewise['Confirmed'],
                name="South Korea",
                line_color='white',
                mode='lines',
                opacity=0.8)
fig.append_trace(trace1, 1, 1)
# fig.add_trace(trace2, 1, 1)
fig.add_trace(trace3, 1, 1)
fig.add_trace(trace4, 1, 1)
fig.add_trace(trace5, 1, 1)
# fig.add_trace(trace6, 1, 1)
fig.add_trace(trace7, 1, 1)
fig.update_layout(title_text = 'CoronaVirus Spead comparison With Rest of World',template="plotly_dark",font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white')   )
fig.show()