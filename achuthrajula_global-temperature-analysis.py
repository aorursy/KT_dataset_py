import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

raw_data = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv")

raw_data.head()

#Removing duplicate country values and Null values



data = raw_data[~raw_data['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



data = data.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



#Calculating country's average temperature



countries = np.unique(data['Country'])

mean_temp = []

for country in countries:

    mean_temp.append(data[data['Country'] == country]['AverageTemperature'].mean())





    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



layout = dict(

    title = 'Average land temperature in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
mean_temp_bar, countries_bar = (list(x) for x in zip(*sorted(zip(mean_temp, countries), 

                                                             reverse = True)))

sns.set(font_scale=0.9) 

f, ax = plt.subplots(figsize=(4.5, 50))

colors_cw = sns.color_palette('coolwarm', len(countries))

sns.barplot(mean_temp_bar, countries_bar, palette = colors_cw[::-1])

Text = ax.set(xlabel='Average temperature', title='Average land temperature in countries')