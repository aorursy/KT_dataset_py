# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train.head()
train.describe()
train.info()
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
train.isnull().sum()
train['Province_State'].fillna(' ',inplace=True)
train.head()
import plotly_express as px
# last date
last_date = train.Date.max()
last_date
# Countries with the most cases till last date
countries = train[train['Date']==last_date]
countries = countries.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()
countries = countries.nlargest(10,'ConfirmedCases')
countries.head()
# Trend for top 10
case_trend = train.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
case_trend = case_trend.merge(countries, on='Country_Region')
case_trend.drop(['ConfirmedCases_y','Fatalities_y'],axis=1, inplace=True)
case_trend.rename(columns={'Country_Region':'Country', 'ConfirmedCases_x':'Cases', 'Fatalities_x':'Deaths'}, inplace=True)

case_trend.head()
px.line(case_trend, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')
px.line(case_trend, x='Date', y='Deaths', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries')
#Add columns for studying logarithmic trends
case_trend['ln(Cases)'] = np.log(case_trend['Cases']+1)# Added 1 to remove error due to log(0).
case_trend['ln(Deaths)'] = np.log(case_trend['Deaths']+1)
px.line(case_trend, x='Date', y='ln(Cases)', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries(Logarithmic Scale)')
px.line(case_trend, x='Date', y='ln(Deaths)', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries(Logarithmic Scale)')
# Mortality Rate
case_trend['Mortality Rate%'] = round((case_trend.Deaths/case_trend.Cases)*100,2)
px.line(case_trend, x='Date', y='Mortality Rate%', color='Country', title='Variation of Mortality Rate% \n(Top 10 worst affected countries)')
df_usa = train.query("Country_Region=='US'")
US_cases = df_usa.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
US_cases.head()
px.bar(US_cases, x='Date', y='ConfirmedCases')
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Bar(name='Cases', x=US_cases['Date'], y=US_cases['ConfirmedCases']),
    go.Bar(name='Deaths', x=US_cases['Date'], y=US_cases['Fatalities'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(USA)')
fig.show()
df_India = train.query("Country_Region=='India'")
India_cases = df_India.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
def add_daily_measures(df):
    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']
    for i in range(1,len(df)):
        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']
        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    return df
India_cases = add_daily_measures(India_cases)
India_cases.head()
fig = go.Figure(data=[
    go.Bar(name='Cases', x=India_cases['Date'], y=India_cases['Daily Cases']),
    go.Bar(name='Deaths', x=India_cases['Date'], y=India_cases['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(India)')
fig.show()
# Give Lockdown Notation
fig.update_layout(barmode='overlay', title='Daily Case and Death count(India)',
                 annotations=[dict(x='2020-03-23', y=106, xref="x", yref="y", text="Lockdown Imposed(23rd March)", showarrow=True, arrowhead=1, ax=-100, ay=-100)])
fig.show()
df_Italy = train.query("Country_Region=='Italy'")
Italy_cases = df_Italy.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
Italy_cases = add_daily_measures(Italy_cases)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=Italy_cases['Date'], y=Italy_cases['Daily Cases']),
    go.Bar(name='Deaths', x=Italy_cases['Date'], y=Italy_cases['Daily Deaths'])
])
# Give Lockdown Notation
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Italy)',
                 annotations=[dict(x='2020-03-09', y=1797, xref="x", yref="y", text="Lockdown Imposed(9th March)", showarrow=True, arrowhead=1, ax=-100, ay=-100)])
fig.show()
#Spain
df_Spain = train.query("Country_Region=='Spain'")
Spain_cases = df_Spain.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
Spain_cases = add_daily_measures(Spain_cases)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=Spain_cases['Date'], y=Spain_cases['Daily Cases']),
    go.Bar(name='Deaths', x=Spain_cases['Date'], y=Spain_cases['Daily Deaths'])
])
# Give Lockdown Notation
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Spain)',
                 annotations=[dict(x='2020-03-15', y=1797, xref="x", yref="y", text="Lockdown Imposed(15th March)", showarrow=True, arrowhead=1, ax=-100, ay=-100)])
fig.show()
Spain_cases.head()
import seaborn as sns
cases = train.groupby('Country_Region')['ConfirmedCases'].sum().reset_index()
cases.head()
fig = px.pie(cases, values='ConfirmedCases', names='Country_Region')
fig.show()
fig = px.line(Spain_cases, x='Date', y='ConfirmedCases')
fig.show()
df_China = train.query("Country_Region=='China'")
China_cases = df_China.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
fig = px.line(China_cases, x='Date', y='ConfirmedCases')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

fig = px.line(India_cases, x='Date', y='ConfirmedCases')
fig.update_xaxes(rangeslider_visible=True)
fig.show()
fig = px.line(US_cases, x='Date', y='ConfirmedCases')
fig.update_xaxes(rangeslider_visible=True)
fig.show()