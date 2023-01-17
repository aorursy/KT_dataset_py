# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()

#print(check_output(["ls", "../input"]).decode("utf8"))

indicators = pd.read_csv("../input/Indicators.csv")

print('Data loaded')
#indicators['CountryCode'].unique()

mortality5_evolution = indicators[indicators['IndicatorCode'] == 'SH.DYN.MORT']

mortality5_evolution.head(20)

world_regions = ('ARB', 'CSS', 'EMU', 'EUU', 'HIC', 'OEC', 'LCN', 'LAC', 

                 'LMY', 'LMC', 'MNA', 'MIC', 'NAC', 'OED', 'PSS', 'SAS', 'WLD')

selected_regions = ('ARB', 'CSS', 'EUU', 'LCN', 

                    'MNA', 'NAC', 'SAS')

drop_indices     = mortality5_evolution['CountryCode'].isin(world_regions)

selected_regions = mortality5_evolution['CountryCode'].isin(selected_regions)



mortality5_by_region = mortality5_evolution[selected_regions == True]

#mortality5_by_country_1960 = mortality5_evolution[(drop_indices == False)

#                                    & (select_1960)]

mortality5_by_region.head()
sns.set_style('ticks')

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns_diag = sns.pointplot(x="Year", y="Value", data=mortality5_by_region, hue="CountryName", 

                         ax=ax, palette=sns.color_palette("muted"))

sns_diag.set(xlabel='Year', ylabel='Mortality rate per 1000 (under 5 years)')

years = list(range(1960,2016))

sns_diag.set_xticklabels(years, rotation=90)

plt.title("Evolution of Mortality Rate Under 5 from 1960 to 2015")

plt.show()
#np.max(mortality5_by_country.Value.values)

#np.max(mortality5_by_country.Year.values)

#mortality5_by_country[mortality5_by_country['Year'] == 2015]

#mortality5_by_country[mortality5_by_country['Value'] > 300]

#mortality5_by_country_2015['CountryName'].unique()
scl = [[0.0, 'rgb(0,245,0)'],   [0.05, 'rgb(0,150, 0)'], [0.1, 'rgb(0,80,80)'],

       [0.15, 'rgb(40,80,80)'], [0.2, 'rgb(200,20,20)'], [1.0, 'rgb(245,0,0)']]

import time

data = dict()

layout = dict()

fig = dict()

for i in range(0,5):

    year_select = mortality5_evolution['Year'] == (1975 + i * 10)

    data[str(i)] = [ dict(

        type           ='choropleth',

        colorscale     = scl,

        autocolorscale = False,

        locations      = mortality5_evolution[(drop_indices == False)

                                             & (year_select)].CountryCode.values,

        z              = mortality5_evolution[(drop_indices == False)

                                             & (year_select)].Value.values,

        zmin           = 0.0,

        zmax           = 444.0,

        text           = mortality5_evolution[(drop_indices == False)

                                             & (year_select)].CountryName,

        marker         = dict(

                           line = dict (

                              color = 'rgb(255,255,255)',

                              width = 2

                         ) ),

        colorbar    = dict(

                        title = "Mortality rate per 1000 ")

                      ) 

              ]



    layout[str(i)] = dict(

        title = '{} in {}'.format(mortality5_evolution[(drop_indices == False)

                                             & (year_select)].IndicatorName.unique()[0],

                                  mortality5_evolution[(drop_indices == False)

                                             & (year_select)].Year.unique()[0]),

        geo = dict(

            scope      = 'world',

            projection = dict( type='Mercator' ),

            showlakes  = True,

            lakecolor  = 'rgb(255, 255, 255)'),

             )  

    fig[str(i)] = dict( data=data[str(i)], layout=layout[str(i)] )

    iplot( fig[str(i)], filename='d3-cloropleth-map' )

    time.sleep(1)