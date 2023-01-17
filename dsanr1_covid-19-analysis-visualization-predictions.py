import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import plotly.graph_objects as go

import plotly.express as px
covid19 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid19.info()
covid19['Active'] = covid19['Confirmed'] - covid19['Deaths'] - covid19['Recovered']
covid19['ObservationDate'] = pd.to_datetime(covid19['ObservationDate'],  format='%m/%d/%Y')
covid19.sample(5)
total_countries = []

for i in covid19['ObservationDate'].unique():

    total_countries.append(covid19[covid19['ObservationDate']==i]['Country/Region'].unique().size)
plt.figure(figsize=(20,5))

plt.plot(covid19['ObservationDate'].unique(), total_countries, label='number of countries')

plt.legend()

plt.xticks(rotation=75)

plt.show()
by_day = covid19.groupby('ObservationDate').sum().sort_values(by='Confirmed')
by_day.tail()
def plot_cases_by_day(df):

    fig, ax1 = plt.subplots(1,1,figsize=(18,7))

    ax1.plot(df['Confirmed'], label='Confirmed')

    ax1.legend(loc='upper left')

    ax1.set_xticklabels(by_day.index, rotation=75)

    ax1.set_ylabel(by_day.columns[1:2][0], fontsize=15, color='b')



    ax2=ax1.twinx()

    ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler

    ax2.plot(df['Deaths'], 'r', label='Deaths')

    ax2.legend(loc='upper center', bbox_to_anchor=(.3, .998))

    ax2.set_ylabel('Deaths', fontsize=15, color='r')



    ax3=ax1.twinx()

    ax3.spines['right'].set_position(('axes', 1.06))

    ax3._get_lines.prop_cycler = ax1._get_lines.prop_cycler

    ax3.plot(df['Recovered'], 'g', label='Recovered')

    ax3.legend(loc='upper right', bbox_to_anchor=(.6, .998))

    ax3.set_ylabel('Recovered', fontsize=15, color='g')

    ax3.set_xticks(df.index)

    
plot_cases_by_day(by_day)
def plot_ratio(df):

    fig, ax1 = plt.subplots(1,1,figsize=(20,7))

    ax1.plot(df['Deaths']/df['Confirmed'], 'r', label='Death Ratio')

    ax1.legend(loc='upper left')

    ax1.set_xticklabels(by_day.index, rotation=75)

    ax1.set_ylabel('Death Ratio', fontsize=15, color='r')



    ax2=ax1.twinx()

    ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler

    ax2.plot(df['Recovered']/df['Confirmed'], 'g', label='Recovered Ratio')

    ax2.legend(loc='upper center')

    ax2.set_ylabel('Recovered Ratio', fontsize=15, color='g')

plot_ratio(by_day)
plt.figure(figsize=(20,7))

plt.bar(by_day.index, by_day['Deaths'], label='Deaths')

plt.bar(by_day.index, by_day['Active'], bottom=by_day['Deaths'], label='Under Treatment')

plt.bar(by_day.index, by_day['Recovered'], bottom = by_day['Confirmed'] - by_day['Recovered'], label='Recovered')

# plt.bar(by_day.index, by_day['Recovered'], bottom=by_day['Deaths'], label='Recovered')

# plt.bar(by_day.index, by_day['Confirmed'] - (by_day['Deaths']+by_day['Recovered']), bottom=by_day['Deaths']+by_day['Recovered'], label='Under Treatment')

plt.legend()

plt.xticks(rotation=75)

plt.show()
def plot_recovery_days(df):

    recovery_days = [0]

    for i in range(df.shape[0]):

        for j in range(recovery_days[-1], df.shape[0]):

            if df.iloc[j]['Deaths'] + df.iloc[j]['Recovered'] > df.iloc[i]['Confirmed']:

                recovery_days.append(j - i)

                break

        else:

            break

#     return recovery_days

    plt.plot(recovery_days[1:])

    plt.ylabel('days')
plot_recovery_days(by_day)
(by_day['Deaths'] / (by_day['Deaths'] + by_day['Recovered'])).tail(40).plot(figsize=(10,5))
china_by_day = covid19[(covid19['Country/Region']=='Mainland China')].groupby('ObservationDate').sum()
china_by_day.tail()
plot_cases_by_day(china_by_day)
plot_ratio(china_by_day)
plot_recovery_days(china_by_day)
(china_by_day['Deaths'] / (china_by_day['Deaths'] + china_by_day['Recovered'])).tail(25).plot(figsize=(10,5))
fig = go.Figure(data=[

    go.Bar(name='Deaths', x=china_by_day.index, y=china_by_day['Deaths']),

    go.Bar(name='Under Treatment', x=china_by_day.index, y=china_by_day['Active']),

    go.Bar(name='Recovered', x=china_by_day.index, y=china_by_day['Recovered'])],

                

    layout=go.Layout(height=500))

fig.update_layout(barmode='stack')

fig.show()
Hubei_by_day = covid19[covid19['Province/State']=='Hubei'].groupby('ObservationDate').sum()
fig = go.Figure(data=[

    go.Bar(name='Deaths', x=Hubei_by_day.index, y=Hubei_by_day['Deaths']),

    go.Bar(name='Under Treatment', x=Hubei_by_day.index, y=Hubei_by_day['Active']),

    go.Bar(name='Recovered', x=Hubei_by_day.index, y=Hubei_by_day['Recovered'])],

                

    layout=go.Layout(height=500))

fig.update_layout(barmode='stack')

fig.show()
plot_cases_by_day(Hubei_by_day)
plot_ratio(Hubei_by_day)
(Hubei_by_day['Deaths'] / (Hubei_by_day['Deaths'] + Hubei_by_day['Recovered'])).tail(25).plot(figsize=(10,5))
Non_hubei = covid19[(covid19['Province/State']!='Hubei') & (covid19['Country/Region']=='Mainland China')].groupby('ObservationDate').sum()
fig = go.Figure(data=[

    go.Bar(name='Deaths', x=Non_hubei.index, y=Non_hubei['Deaths']),

    go.Bar(name='Under Treatment', x=Non_hubei.index, y=Non_hubei['Active']),

    go.Bar(name='Recovered', x=Non_hubei.index, y=Non_hubei['Recovered'])],

                

    layout=go.Layout(height=500))

fig.update_layout(barmode='stack')

fig.show()
plot_cases_by_day(Non_hubei)
plot_ratio(Non_hubei)
(Non_hubei['Deaths'] / (Non_hubei['Deaths'] + Non_hubei['Recovered'])).tail(25).plot(figsize=(10,5))
plot_recovery_days(Non_hubei)
plt.figure(figsize=(15,5))

plt.plot(Non_hubei.index, Non_hubei['Confirmed'].diff() / Non_hubei['Confirmed'].diff().shift(1)[:30])

plt.plot(Non_hubei.index, [1 for i in range(len(Non_hubei.index))])
plt.figure(figsize=(15,5))

plt.plot((Non_hubei['Confirmed'].diff(2) / Non_hubei['Confirmed'].diff(2).shift(1))[:30])

plt.plot(Non_hubei.index, [1 for i in range(len(Non_hubei.index))])
Non_hubei.loc['02/05/2020', 'Confirmed']
Non_china = covid19[(covid19['Country/Region']!='Mainland China')].groupby('ObservationDate').sum()
Non_china.tail()
plot_cases_by_day(Non_china)
plot_ratio(Non_china)
plot_recovery_days(Non_china)
(Non_china['Deaths'] / (Non_china['Deaths'] + Non_china['Recovered'])).plot(figsize=(10,5))
plt.figure(figsize=(15,5))

plt.plot(Non_china['Confirmed'].diff() / Non_china['Confirmed'].diff().shift(1))

plt.plot(Non_china.index, [1 for i in range(len(Non_china.index))])
plt.figure(figsize=(15,5))

plt.plot(Non_china['Confirmed'].diff(3) / Non_china['Confirmed'].diff(3).shift())

plt.plot(Non_china.index, [1 for i in range(len(Non_china.index))])
by_country = covid19[covid19['ObservationDate']==covid19['ObservationDate'].unique()[-1]].groupby('Country/Region').sum().iloc[:, 1:].sort_values(by='Confirmed', ascending=False)
by_country['Case Fatality Rate'] = by_country['Deaths'] / by_country['Confirmed']
by_country = by_country.reset_index()
px.bar(by_country[:30], x = 'Country/Region', y = 'Confirmed', log_y=True, height=500)
px.bar(by_country.sort_values(by='Active', ascending=False)[:30], x = 'Country/Region', y = 'Active', log_y=True, height=500)
px.bar(by_country.sort_values(by='Deaths', ascending=False)[:30], x = 'Country/Region', y = 'Deaths', height=500)
px.bar(by_country[by_country['Confirmed']>100].sort_values(by='Case Fatality Rate', ascending=False).iloc[:40, :], x = 'Country/Region', y = 'Case Fatality Rate', height=500)
by_country = by_country.set_index('Country/Region')
death_ratio = (by_country['Deaths']/(by_country['Deaths'] + by_country['Recovered']))[((by_country['Recovered'])>20)].sort_values(ascending=False)
px.bar(x=death_ratio.index, y=death_ratio.values)
data = go.Choropleth(z = by_country['Confirmed'], locations = by_country.index, locationmode = 'country names', text = 'Confirmed', colorscale = 'YlOrRd', 

        reversescale=False, marker_line_color='darkgray', marker_line_width=0.5, colorbar_tickprefix = '', colorbar_title = 'cases')

    

layout = go.Layout(autosize=False, width=1000, height=500, title_text='Confirmed Cases',

        geo=dict(showframe=True, showcoastlines=True, projection_type='robinson'))



fig = go.Figure(data = data, layout = layout)



fig.show()
data = go.Choropleth(z = by_country['Deaths'], locations = by_country.index, locationmode = 'country names', text = 'Deaths', colorscale = 'Reds', autocolorscale=False,

        reversescale=False, marker_line_color='darkgray', marker_line_width=0.5, colorbar_tickprefix = '', colorbar_title = 'Deaths')

    

layout = go.Layout(autosize=False, width=1000, height=500, title_text='Deaths',

        geo=dict(showframe=True, showcoastlines=True, projection_type='robinson'))



fig = go.Figure(data = data, layout = layout)



fig.show()
covid19[covid19['ObservationDate']==covid19['ObservationDate'].unique()[-1]].groupby(['Country/Region', 'Province/State']).sum().groupby(['Country/Region']).size()
USA = covid19[covid19['ObservationDate']==covid19['ObservationDate'].unique()[-1]].groupby(['Country/Region','Province/State']).sum().xs('US').iloc[:, 1:].sort_values('Confirmed', ascending=False)
px.bar(USA, x=USA.index, y='Confirmed')
fig = px.line(covid19[(covid19['Country/Region']!='Mainland China')].groupby(['ObservationDate', 'Country/Region']).sum().reset_index().iloc[1000:, :],

              x="ObservationDate", y="Confirmed", color='Country/Region')

fig.show()
df = covid19.groupby(['Country/Region','ObservationDate']).sum()
country_100 = []

for i in by_country.index[by_country['Confirmed'] >= 100]:

    country_100.append(pd.Series(df[df['Confirmed'] > 100].loc[i].reset_index()['Confirmed'], name=i))
country_100 = pd.concat(country_100, axis=1)
country_100
px.line(pd.melt(country_100.reset_index(), id_vars='index', var_name='Country', value_name='Confirmed Cases'), x='index', y='Confirmed Cases', color='Country', log_y=True)
South_Korea = covid19[covid19['Country/Region']=='South Korea'].groupby('ObservationDate').sum()
fig = go.Figure(data=[

    go.Bar(name='Deaths', x=South_Korea.index, y=South_Korea['Deaths']),

    go.Bar(name='Under Treatment', x=South_Korea.index, y=South_Korea['Active']),

    go.Bar(name='Recovered', x=South_Korea.index, y=South_Korea['Recovered'])],

                

    layout=go.Layout(height=500))

fig.update_layout(barmode='stack')

fig.show()
plt.figure(figsize=(15,5))

plt.plot((South_Korea['Confirmed'].diff() / South_Korea['Confirmed'].diff().shift(1))[35:])

plt.plot(South_Korea.index, [1 for i in range(len(South_Korea.index))])
Italy = covid19[covid19['Country/Region']=='Italy'].groupby('ObservationDate').sum()
fig = go.Figure(data=[

    go.Bar(name='Deaths', x=Italy.index, y=Italy['Deaths']),

    go.Bar(name='Under Treatment', x=Italy.index, y=Italy['Active']),

    go.Bar(name='Recovered', x=Italy.index, y=Italy['Recovered'])],

                

    layout=go.Layout(height=500))

fig.update_layout(barmode='stack')

fig.show()
plt.figure(figsize=(15,5))

plt.plot(Italy['Confirmed'].diff(2) / Italy['Confirmed'].diff(2).shift()[30:])

plt.plot(Italy.index, [1 for i in range(len(Italy.index))])
India = covid19[covid19['Country/Region']=='India'].groupby('ObservationDate').sum()
fig = go.Figure(data=[

    go.Bar(name='Deaths', x=India.index, y=India['Deaths']),

    go.Bar(name='Under Treatment', x=India.index, y=India['Active']),

    go.Bar(name='Recovered', x=India.index, y=India['Recovered'])],

                

    layout=go.Layout(height=500))

fig.update_layout(barmode='stack')

fig.show()
plt.figure(figsize=(15,5))

plt.plot(India['Confirmed'].diff(2) / India['Confirmed'].diff(2).shift(1)[35:])

plt.plot(India.index, [1 for i in range(len(India.index))])
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
df.isnull().sum()
df[(df['age']>0) & (df['age']<=65) & (df['death']=='1')].shape[0] / df[(df['age']>0) & (df['age']<=65)].shape[0]
df[(df['age']>55) & (df['age']<=60) & (df['death']=='1')].shape[0] / df[(df['age']>55) & (df['age']<=60)].shape[0]
df[(df['age']>60) & (df['age']<=65) & (df['death']=='1')].shape[0] / df[(df['age']>60) & (df['age']<=65)].shape[0]
df[(df['age']>65) & (df['death']=='1')].shape[0] / df[df['age']>65].shape[0]
df[(df['age']>75) & (df['death']=='1')].shape[0] / df[df['age']>75].shape[0]
df['age'].sort_values().reset_index()['age'].plot(kind='hist')
df[df['death']=='1']['age'].sort_values().reset_index()['age'].plot(kind='hist')