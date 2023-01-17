import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px

import re

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update', 'ObservationDate'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df['Country'].replace({'Mainland China': 'China', ' Azerbaijan': 'Azerbaijan', '(\'St. Martin\',)':'St. Martin', 'occupied Palestinian territory':'Palestine', 'Gambia, The':'The Gambia'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_countries = pd.read_csv('../input/countries-of-the-world/countries of the world.csv', decimal=',')

df_countries['Country'] = df_countries['Country'].str.strip() 

df_countries = df_countries.loc[:, df_countries.columns.intersection(['Country', 'Population'])]



conversion_dict = {'Korea, South': 'South Korea', 'Korea, North': 'North Korea', 'Bahamas, The': 'The Bahamas', 'United States': 'US', 'United Kingdom': 'UK', 'Gambia, The': 'The Gambia', 'Trinidad & Tobago': 'Trinidad and Tobago', 'Congo, Dem. Rep.': 'Congo', 'Cote d\'Ivoire': 'Ivory Coast', 'Congo, Repub. of the':'Republic of the Congo', 'Bosnia & Herzegovina': 'Bosnia and Herzegovina', 'West Bank':'Palestine', }

df_countries['Country'] = df_countries['Country'].replace(conversion_dict)





df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df_change_day = df.groupby('Date').sum().reset_index()

#df_change_day['Growth Rate'] = df_change_day["Confirmed"].pct_change(fill_method='ffill') + 1 

df_change_day.tail(10)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_change_day['Date'], 

                         y=df_change_day['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='rgb(40, 146, 215)', width=2)

                        ))

fig.add_trace(go.Scatter(x=df_change_day['Date'], 

                         y=df_change_day['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='rgb(27, 64, 109)', width=2)

                        ))

fig.add_trace(go.Scatter(x=df_change_day['Date'], 

                         y=df_change_day['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='rgb(17, 86, 142)', width=2)

                        ))





fig.update_layout(

    title='Worldwide Corona Virus Cases',

    xaxis_tickfont_size=14,

    plot_bgcolor='rgb(245,245,245)',

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
df_change_country = df.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

df_change_country = pd.merge(df_change_country, df_countries, on='Country', how='inner').sort_values(by=['Country', 'Date']) # Change to left join and run the query below to see any missed countries

#[x for x in df_change_country[df_change_country.isnull()['Population'] == True]['Country'].unique()]





df_change_country['Confirmed Ratio'] = df_change_country['Confirmed']/df_change_country['Population']  

#df_change_country['Deaths Ratio'] = df_change_country['Deaths']/df_change_country['Population']  

#df_change_country['Recovered Ratio'] = df_change_country['Recovered']/df_change_country['Population']  





df_change_country['Growth Rate'] = df_change_country.groupby(['Country'])['Confirmed'].pct_change(fill_method='ffill') + 1 

#df_change_country['Deaths Multiplier'] = df_change_country.groupby(['Country'])['Deaths'].pct_change()

#df_change_country['Recovered Multiplier'] = df_change_country.groupby(['Country'])['Recovered'].pct_change()



#df_change_country

df_country = df.groupby(['Country'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index().sort_values(['Confirmed'], ascending=False)

df_country.head(10)
k = 10

list_of_countries = [x for x in df_country['Country'].head(k)]



fig = go.Figure()



for i in range(k):

    fig.add_trace(go.Scatter(x=df_change_country[df_change_country['Country'] == list_of_countries[i]]['Date'], 

                         y=df_change_country[df_change_country['Country'] == list_of_countries[i]]['Confirmed'],

                         mode='lines+markers', 

                         name=list_of_countries[i])

                        )



fig.update_layout(

    title='Confirmed Corona Virus Cases - Top {0} Countries'.format(str(k)),

    xaxis_tickfont_size=14,

    plot_bgcolor='rgb(245,245,245)',

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

        type='log'

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
df_change_country[df_change_country["Country"] == 'Switzerland']
k = 10

list_of_countries = [x for x in df_country['Country'].head(k).unique()]



df_dist_country = df_change_country[df_change_country['Growth Rate'].replace([np.inf, -np.inf], np.nan).notnull()][["Country","Growth Rate"]]

#df_dist_country = df_dist_country[df_dist_country["Country"] in df_dist_country]





#df_dist_country

booleans = []

for result in df_dist_country['Country']:

    if result in list_of_countries:

        booleans.append(True)

    else:

        booleans.append(False)

        

df_dist_country = df_dist_country[booleans]

#df_dist_country = df_change_country[df_change_country["Country"] == 'Switzerland']



#df_dist_country

f, ax = plt.subplots(figsize=(18, 6))

ax = sns.violinplot(x="Country", y="Growth Rate", data=df_dist_country, inner='stick', palette='pastel',scale='count',bw=0.2)

ax.set_title('Growth Rate Distribution - Top 10 Countries', fontdict={'fontsize': '16'})

plt.grid()

ax.set_yticks(np.arange(0, 3, 0.1)) 

ax.yaxis.grid(True, linestyle='-')

ax.xaxis.grid(False) 



plt.rcParams['axes.facecolor'] = '#F4F4F4'



#plt.axhline(0, linewidth = 0.5, linestyle=':', color='#0A0A0A')

#plt.axhline(1, linewidth = 0.5, linestyle=':', color='#0A0A0A')

ax.set(ylim=(1, 3))

plt.show()

cty = 'Switzerland'

df_cty = df_change_country[df_change_country['Country'] == cty]

df_cty
for prog, ax in zip(prog_list, axes.flatten()[:5]):

    scores = df.loc[(df['prog'] == prog)]['score']



    # note how I put 'ax' here

    sns.distplot(scores, norm_hist=True, ax=ax, color='b')



    # change all the axes into ax

    sigma = round(scores.std(), 3)

    mu = round(scores.mean(), 2)

    ax.set_xlim(1,7)

    ax.set_xticks(range(2,8))

    ax.set_xlabel('Score - Mean: {} (σ {})'.format(mu, sigma))

    ax.set_ylabel('Density')



plt.show()
k = 10

list_of_countries = [x for x in df_country['Country'].head(k).unique()]



f, axes = plt.subplots(int(k/5), int(k/2), figsize=(22, 5), sharex=True)



for i in range(k):

    if i < k/2:

        sns.distplot(df_change_country[df_change_country['Country'] == list_of_countries[i]]["Growth Rate"], color="b", ax=axes[0, i])

        axes[0, i].label = 'abc'

    else:

        sns.distplot(df_change_country[df_change_country['Country'] == list_of_countries[1]]["Growth Rate"], color="b", ax=axes[1, i- int(k/2)])



plt.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_cty['Date'], 

                         y=df_cty['Growth Rate'],

                         mode='lines+markers',

                         name='Growth Rate',

                         line=dict(color='rgb(28, 26, 30)', width=2)

                        ))

fig.add_trace(go.Scatter(x=df_cty['Date'],  

                         y=df_cty['Growth Rate'].rolling(window=10).mean(),

                         mode='lines',

                         name='Moving Average (10) Growth Rate',

                         line=dict(color='rgba(62, 13, 19, 0.8)', width=2)

                        ))

fig.add_trace(go.Scatter(x=df_cty['Date'],  

                         y=df_cty['Growth Rate'].rolling(window=5).mean(),

                         mode='lines',

                         name='Moving Average (5) Growth Rate',

                         line=dict(color='rgba(102, 16, 23, 0.8)', width=2)

                        ))

fig.add_trace(go.Scatter(x=df_cty['Date'],  

                         y=[df_cty['Growth Rate'].mean()] * len(df_cty['Date']),

                         mode='lines',

                         name='Fixed Average Growth Rate',

                         line=dict(color='rgba(153, 88, 42, 0.8)', width=2)

                        ))



fig.update_layout(

    title=cty + ' Growth Rate',

    xaxis_tickfont_size=14,

    plot_bgcolor='rgb(245,245,245)',

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
df_prophet_confirmed = df_cty[['Date','Confirmed']] 



df_prophet_confirmed.columns = ['ds','y'] 

df_prophet_confirmed['ds'] = pd.to_datetime(df_prophet_confirmed['ds'])

df_prophet_confirmed['cap'] = 10000

df_prophet_confirmed['floor'] = 0



m = Prophet(interval_width=0.95)



m.fit(df_prophet_confirmed)

future = m.make_future_dataframe(periods=7) 

future['floor'] = 0



forecast = m.predict(future)

#forecast.tail().T



confirmed_forecast_plot = m.plot(forecast)

m.plot_components(forecast);