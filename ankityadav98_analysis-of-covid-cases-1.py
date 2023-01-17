import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
confirmed_global= pd.read_csv('/kaggle/input/covid19-assignment/time_series_covid19_confirmed_global.csv')

deaths_global= pd.read_csv('/kaggle/input/covid19-assignment/time_series_covid19_deaths_global.csv')

recovered_global= pd.read_csv('/kaggle/input/covid19-assignment/time_series_covid19_recovered_global.csv')
confirmed_global.head()
confirmed_global.columns
dates= confirmed_global.columns[4:]

dates
confirmed_df = confirmed_global.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],

                                     value_vars=dates, var_name='Date', value_name='Confirmed'

                                    )
death_df= deaths_global.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],

                                     value_vars=dates, var_name='Date', value_name='Death')
recovered_df= recovered_global.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],

                                     value_vars=dates, var_name='Date', value_name='Recovered')
recovered_df.head()
death_df.head()
confirmed_df.head()
confirmed_df.info()
all_data = confirmed_df.merge(right=death_df, how='left',

                              on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])



# Merging all_data and recovered_df

all_data = all_data.merge(right=recovered_df, how='left',

                            on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
all_data.head()
all_data.info()
all_data['Date']=pd.to_datetime(all_data['Date'])
all_data.isna().sum()
plt.figure(figsize=(9,6))

sns.heatmap(all_data.isna(),yticklabels=False,cmap='viridis',cbar=False)
#filling null values by 0.

all_data['Recovered']= all_data['Recovered'].fillna(0)
all_data['Recovered']= all_data['Recovered'].astype('int64')
all_data['Active']= all_data['Confirmed']- all_data['Recovered']- all_data['Death']
all_data.head()
all_data.loc[all_data['Province/State']=='Greenland','Country/Region']='Greenland'
data_grouped = all_data.groupby(['Date', 'Country/Region'])['Confirmed','Death','Recovered','Active'].sum().reset_index()
data_grouped.info()
data_grouped.head()
country_wise= data_grouped.groupby(['Country/Region'])['Confirmed','Death','Recovered','Active'].sum().reset_index()
country_wise.head()
country_wise.info()
day_wise= data_grouped.groupby(['Date'])['Confirmed','Death','Recovered','Active'].sum().reset_index()
day_wise.tail()
line_plot = day_wise[['Date','Confirmed','Recovered', 'Active','Death']]

line_plot = line_plot.melt(id_vars='Date', value_vars=['Confirmed','Recovered', 'Active','Death'],var_name='Status', value_name='No. of cases')

line_plot.head()
px.line(line_plot, x='Date', y='No. of cases', color='Status',title='Total Cases in the World upto 13-07-2020')
confirm_color= 'deepskyblue'

death_color='red'

recovered_color='limegreen'

active_color='grey'
def bar_plot(df,col,color):

    fig=px.bar(df,x='Date',y=col,color_discrete_sequence=[color],title=' Total '+str(col)+' cases in the World')

    fig.show()
bar_plot(day_wise,'Confirmed',confirm_color)
bar_plot(day_wise,'Death',death_color)
bar_plot(day_wise,'Recovered',recovered_color)
bar_plot(day_wise,'Active',active_color)
px.line(line_plot, x='Date', y='No. of cases',log_y=True, color='Status',title='Total Cases in the World on Logarthmic Scale')
def plot_bar_countrywise(col,country, color):

    df=all_data[all_data['Country/Region']==country]

    df['prev_case']=df[col].shift(1,axis=0)

    df['daily_'+str(col)]= df[col]-df['prev_case']

    fig = px.bar(df, x="Date", y=df['daily_'+str(col)], width=700, color_discrete_sequence=[color])

    fig.update_layout(title='Daily Increase in '+col+' Cases in {}'.format(country), yaxis_title="No. of Cases", xaxis_title="Date")

    fig.show()
plot_bar_countrywise('Confirmed','India',confirm_color)
plot_bar_countrywise('Death','India',death_color)
plot_bar_countrywise('Recovered','India',recovered_color)
plot_bar_countrywise('Confirmed','US',confirm_color)
plot_bar_countrywise('Death','US',death_color)
plot_bar_countrywise('Recovered','US',recovered_color)
plot_bar_countrywise('Confirmed','Brazil',confirm_color)
plot_bar_countrywise('Death','US',death_color)
plot_bar_countrywise('Recovered','US',recovered_color)
countrywise_daywise_data=all_data[['Country/Region','Date','Confirmed','Recovered', 'Active','Death']]

countrywise_daywise_data=countrywise_daywise_data.melt(id_vars=['Date','Country/Region'], value_vars=['Confirmed','Recovered', 'Active','Death'],var_name='Status', value_name='No. of cases')

countrywise_daywise_data.head()
def line_plot_daywise(df,country):

    new_df=df[df['Country/Region']==country].reset_index(drop=True)

    fig=px.line(new_df, x='Date', y='No. of cases', color='Status',title='Total Cases in {} upto 13-07-2020 (Linear scale)'.format(country))

    fig.show()
line_plot_daywise(countrywise_daywise_data,'India')
line_plot_daywise(countrywise_daywise_data,'US')
line_plot_daywise(countrywise_daywise_data,'Spain')
line_plot_daywise(countrywise_daywise_data,'Brazil')
def plot_rec_death_rate(df,country):

    new_df= df[df['Country/Region']==country].reset_index(drop=True)

    new_df['recovery_rate']=new_df['Recovered']/new_df['Confirmed']

    new_df['CFR']= new_df['Death']/new_df['Confirmed']

    new_df.fillna(0,inplace=True)

    plt.figure(figsize=(8,6))

    plt.plot(new_df['Date'],new_df['recovery_rate'],label='Recovery Rate')

    plt.plot(new_df['Date'],new_df['CFR'],label='Case Fatality Ratio')

    plt.title(label='Recovery Rate and CFR in {}'.format(country))

    plt.xlabel('Time')

    plt.xticks(rotation=90)

    plt.ylabel('Ratio')

    plt.ylim(0,1)

    plt.legend()

    plt.show()
plot_rec_death_rate(data_grouped,'India')
plot_rec_death_rate(data_grouped,'China')
plot_rec_death_rate(data_grouped,'US')
plot_rec_death_rate(data_grouped,'Spain')
#Total cases as on 13/07/2020

cases=[13104391,573003,7161007,5370381]

total_cases = pd.DataFrame(cases, index=['Confirmed','Death','Recovered','Active'], columns=['No. of Cases'])
plt.figure(figsize=(8,8))

plt.pie(total_cases['No. of Cases'],labels=total_cases['No. of Cases'].index,autopct='%.2f%%',labeldistance=None);

plt.title('Percentage of cases upto 2020-07-13 (Worldwide)',fontdict={'fontsize':18});

plt.legend();