import scipy.stats

import pylab



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio



init_notebook_mode(connected=True) 

pio.templates.default = "plotly_white"



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt 

import statsmodels.graphics.tsaplots as sgt 

import statsmodels.tsa.stattools as sts 

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns

sns.set()
import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
query_str = 'Country_Region == "Indonesia" or Country_Region == "Singapore" or Country_Region == "Philippines" or Country_Region == "Malaysia" or Country_Region == "Thailand"'
maps = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

maps = maps.query(query_str)

maps['Date'] = maps['Date'].astype(str)

maps = maps.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()

maps.tail()
import pycountry



d = dict()



def get_iso3_util(country_name):

    try:

        country = pycountry.countries.get(name=country_name)

        return country.alpha_3

    except:

        country = pycountry.countries.search_fuzzy(country_name)

        return country[0].alpha_3



def get_iso3(country):

    if country in d:

        return d[country]

    else:

        d[country] = get_iso3_util(country)
maps['iso_alpha'] = maps.apply(lambda x: get_iso3(x['Country_Region']), axis=1)

maps['ln(ConfirmedCases)'] = np.log(maps.ConfirmedCases + 1)

maps['ln(Fatalities)'] = np.log(maps.Fatalities + 1)
fig = px.choropleth(

        maps,

        title='Total Confirmed Cases Growth(Logarithmic Scale)',

        locations="iso_alpha",

        color="ln(ConfirmedCases)",

        hover_name="Country_Region", 

        hover_data=["ConfirmedCases"],

        animation_frame="Date",

        scope="asia",

        color_continuous_scale=px.colors.sequential.YlOrRd

)



fig.update_geos(fitbounds="locations", visible=True)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.choropleth(

        maps,

        title='Total Fatality Cases Growth(Logarithmic Scale)',

        locations="iso_alpha",

        color="ln(Fatalities)",

        hover_name="Country_Region", 

        hover_data=["Fatalities"],

        animation_frame="Date",

        scope="asia",

        color_continuous_scale=px.colors.sequential.YlOrRd

)



fig.update_geos(fitbounds="locations", visible=True)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
maps['Mortality Rate%'] = round((maps.Fatalities/maps.ConfirmedCases) * 100, 2)
px.line(

    maps,

    x='Date',

    y='Mortality Rate%',

    color='Country_Region',

    title='Variation of Mortality Rate% in ASEAN'

)
def add_daily_measures(df):

    

    idxs = df.index

    

    df.loc[idxs[0],'Daily Cases'] = df.loc[idxs[0],'ConfirmedCases']

    df.loc[idxs[0],'Daily Deaths'] = df.loc[idxs[0],'Fatalities']

    

    for i in range(1, len(df.index)):

        df.loc[idxs[i],'Daily Cases'] = df.loc[idxs[i],'ConfirmedCases'] - df.loc[idxs[i-1],'ConfirmedCases']

        df.loc[idxs[i],'Daily Deaths'] = df.loc[idxs[i],'Fatalities'] - df.loc[idxs[i-1],'Fatalities']

    

    df.loc[idxs[0],'Daily Cases'] = 0

    df.loc[idxs[0],'Daily Deaths'] = 0

    

    return df
indonesia = maps.query("Country_Region=='Indonesia' and ConfirmedCases > 0")
indonesia = add_daily_measures(indonesia)



fig = go.Figure(data=[

    go.Bar(name='Cases', x=indonesia['Date'], y=indonesia['Daily Cases']),

    go.Bar(name='Deaths', x=indonesia['Date'], y=indonesia['Daily Deaths'])

])





fig.update_layout(barmode='overlay', title='Daily Case and Death count(Indonesia)')

fig.show()