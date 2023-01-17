# Importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
# Importing the Dataset

df = pd.read_csv("../input/all_energy_statistics.csv")

df.head()
# Some attributes

countries = df['country_or_area'].unique() # All countries

all_commodities = df['commodity_transaction'].unique().tolist()

all_categories = df['category'].unique().tolist()

years = list(range(1990, 2015)) # Years
# Function that selects all the commodities with some key

def commodities_select(key):

    commodities = [s for s in all_commodities if key in s]

    return commodities



# Function that selects all categories with some key

def category_select(key):

    key = key.lower()

    category = [s for s in all_categories if key in s]

    return category



# Selecting to the Wind Category

wind = category_select("Wind")

dfWind = df[df['category'].isin(wind)] # Wind DataFrame
# Calculating the mean production

mean_qtd = []

country_wind_producers = []

for country in countries:

    mean = dfWind[dfWind['country_or_area'] == country]['quantity'].mean()

    if(np.isnan(mean) == False and mean > 1500):

        mean_qtd.append(mean)

        country_wind_producers.append(country)



# Visualization

data = [go.Bar(x=mean_qtd, 

               y=country_wind_producers,

               orientation='h')]



layout = go.Layout(xaxis=dict(title="Mean quantity in millions of kwh"),

                   yaxis=dict(title="Countries"),

                  title="Mean Quantity of Wind Energy produced per country")



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename="mean_quantity")

plt.show()
# Leaving the average and analysing each year

data = []

main_countries=['United States', 'United Kingdom', 'Canada', 'China', 'Japan', 'Italy', 'France']

for country in main_countries:

    countryDF = dfWind[dfWind['country_or_area']==country]

    qtd_plot = []

    effective_year = []

    for year in years:

        qtd = countryDF[countryDF['year'] == year]['quantity'].values

        if qtd:

            qtd_plot.append(qtd[0])

            effective_year.append(year)

    # Preparing the plot

    trace = go.Scatter(x=effective_year,

                      y = qtd_plot,

                      name=country,

                      mode='lines+markers')

    data.append(trace)



layout = go.Layout(title="Timeline of Each Country",

                   xaxis=dict(title='year'),

                  yaxis=dict(title='Quantity of Wind Energy - MKwh'))

fig = go.Figure(data=data, layout=layout)

    

py.iplot(data)



plt.show()