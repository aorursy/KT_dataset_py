import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet # https://github.com/facebook/prophet
#From Kaggle Data

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
df_japan = df[df["Country"] == "Japan"].copy()
print(df_japan.tail())
# Only Japan Data

# From https://www.mhlw.go.jp/stf/covid-19/open-data.html by the Ministry of Health, Labor and Welfare

df2_pcr_tested = pd.read_csv('https://www.mhlw.go.jp/content/pcr_tested_daily.csv')

df2_pcr_positive = pd.read_csv('https://www.mhlw.go.jp/content/pcr_positive_daily.csv')



# print(df2_pcr_tested)

# print(df2_pcr_positive)



df2 = df2_pcr_tested.merge(df2_pcr_positive,  on='日付')
print(df2.tail())
pcr_tested_japan = df2

pcr_tested_japan = pcr_tested_japan.rename(columns={'日付': 'Date'})

pcr_tested_japan = pcr_tested_japan.rename(columns={'PCR 検査実施件数(単日)': 'PCR_tested'})

pcr_tested_japan = pcr_tested_japan.rename(columns={'PCR 検査陽性者数(単日)': 'PCR_confirmed'})
fig = go.Figure()

fig.add_trace(go.Bar(x=pcr_tested_japan['Date'],

                y=pcr_tested_japan['PCR_tested'],

                name='PCR_tested',

                marker_color='Blue'

                ))

fig.add_trace(go.Bar(x=pcr_tested_japan['Date'],

                y=pcr_tested_japan['PCR_confirmed'],

                name='PCR_comfirmed',

                marker_color='Red'

                ))

fig.update_layout(

    title='Japan Corona Virus Cases - PCR tested, PCR comfirmed(single day basis)',

    xaxis_tickfont_size=14,

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

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
ctrj_num = (pcr_tested_japan['PCR_confirmed'] / pcr_tested_japan['PCR_tested']) * 100

comfirmed_tested_ratio_japan = pd.DataFrame({'Date': pcr_tested_japan['Date'], 'comfirmed_tested_ratio_japan': ctrj_num})

# Drop Date:2020/4/12

comfirmed_tested_ratio_japan = comfirmed_tested_ratio_japan[comfirmed_tested_ratio_japan["Date"] != "2020/4/12"]

comfirmed_tested_ratio_japan
fig = go.Figure()



fig.add_trace(go.Scatter(x=comfirmed_tested_ratio_japan['Date'],

                y=comfirmed_tested_ratio_japan['comfirmed_tested_ratio_japan'],

                mode='lines+markers',

                name='PCR_comfirmed / PCR_tested Ratio(%)'

                ))

fig.update_layout(

    title='Japan Corona Virus Cases - PCR_comfirmed / PCR_tested Ratio(%)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='% of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
confirmed_japan = df_japan.groupby('Date').sum()['Confirmed'].reset_index()

deaths_japan = df_japan.groupby('Date').sum()['Deaths'].reset_index()

recovered_japan = df_japan.groupby('Date').sum()['Recovered'].reset_index()
ccj_date = confirmed_japan['Date']

ccj_num = confirmed_japan['Confirmed'] - deaths_japan['Deaths'] - recovered_japan['Recovered']

current_confirmed_japan = pd.DataFrame({'Date': ccj_date, 'Current_Confirmed': ccj_num})

# print(current_confirmed_japan)
fig = go.Figure()

fig.add_trace(go.Bar(x=confirmed_japan['Date'],

                y=confirmed_japan['Confirmed'],

                name='Confirmed',

                marker_color='Blue'

                ))

fig.add_trace(go.Bar(x=deaths_japan['Date'],

                y=deaths_japan['Deaths'],

                name='Deaths',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=recovered_japan['Date'],

                y=recovered_japan['Recovered'],

                name='Recovered',

                marker_color='Green'

                ))

fig.add_trace(go.Scatter(x=current_confirmed_japan['Date'],

                y=current_confirmed_japan['Current_Confirmed'],

                mode='lines+markers',

                name='Current Confirmed'

                ))

fig.update_layout(

    title='Japan Corona Virus Cases - Confirmed, Deaths, Recovered, Current Confirmed',

    xaxis_tickfont_size=14,

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

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
confirmed = confirmed_japan

deaths = deaths_japan

recovered = recovered_japan

current_confirmed = current_confirmed_japan
current_confirmed.columns = ['ds','y']

current_confirmed['ds'] = pd.to_datetime(current_confirmed['ds'])
m = Prophet(interval_width=0.95)

m.fit(current_confirmed)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
current_confirmed_forecast_plot = m.plot(forecast)