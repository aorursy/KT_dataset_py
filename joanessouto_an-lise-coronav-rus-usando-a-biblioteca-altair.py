import pandas as pd

import numpy as np

import altair as alt
covid_19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/03-27-2020.csv")
covid_19.head()
covid_19.describe()
covid_19.info()
covid_19.isnull().sum()
map = alt.topo_feature("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json", "countries")
background = alt.Chart(map).mark_geoshape(

    fill='lightblue',

    stroke='white'

).properties(

    width=1000,

    height=600

).project('naturalEarth1')



points = alt.Chart(covid_19.loc[covid_19['Confirmed'] > 0]).mark_circle().encode(

    longitude='Long_:Q',

    latitude='Lat:Q',

    tooltip=['Country_Region:N','Province_State:N','Confirmed:Q'],

    color=alt.value('#FF0000'),

).properties(

    title='Numero de pessoas confirmadas com o Covid19 em diferentes regiões'

)



background + points
np.sum(covid_19[['Confirmed','Deaths','Recovered']]).plot(kind='bar')
import altair as alt



alt.Chart(covid_19.loc[covid_19['Deaths'] > 0]).mark_bar().encode(

   x = 'Country_Region:N',

   y = 'Deaths:Q',

   color = 'Country_Region:N'

)
alt.Chart(covid_19.loc[covid_19['Recovered'] > 0]).mark_bar().encode(

   x = 'Country_Region:N',

   y = 'Recovered:Q',

   color = 'Country_Region:N'

)
import altair as alt



alt.Chart(covid_19).mark_circle().encode(

    x='Deaths:Q',

    y='Confirmed:Q',

    color='Country_Region:N',

    tooltip='Country_Region:N'

).interactive()
confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
confirmed.head()
confirmed.shape
confirmed = np.sum(confirmed.iloc[: ,4 : confirmed.shape[1]])
confirmed.plot()
covid19_series_Deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
covid19_series_Deaths.head()
covid19_series_Deaths.shape
number_deaths = np.sum(covid19_series_Deaths.iloc[:, 4:covid19_series_Deaths.shape[1]])

number_deaths
number_deaths.plot()
mortality_rate = (number_deaths/confirmed)*100

mortality_rate
mortality_rate.plot()
recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
recovered.head()
recovered = np.sum(recovered.iloc[:, 4: recovered.shape[1]])
recovered.plot()
import altair as alt



brush = alt.selection(type='interval')



points = alt.Chart(covid_19.loc[covid_19['Deaths'] > 0]).mark_point().encode(

    x='Country_Region:N',

    y='Deaths:Q',

    color=alt.condition(brush, 'Country_Region:N', alt.value('lightgray'))

).add_selection(

    brush

)



bars = alt.Chart(covid_19.loc[covid_19['Deaths'] > 0]).mark_bar().encode(

   x = 'Country_Region:N',

   y = 'Deaths:Q',

   color = 'Country_Region:N'

).transform_filter(

    brush

)



points & bars