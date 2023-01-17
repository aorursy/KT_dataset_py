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

print('Data loaded now')

#indicators['CountryCode'].unique()

listedCompanies_evolution = indicators[indicators['IndicatorCode'] == 'CM.MKT.LDOM.NO'  ]

listedCompanies_evolution.head(20)

world_regions = ('ARB', 'CSS', 'EMU', 'EUU', 'HIC', 'OEC', 'LCN', 'LAC', 

                 'LMY', 'LMC', 'MNA', 'MIC', 'NAC', 'OED', 'PSS', 'SAS', 'WLD')

selected_countries = (  'MEX', 'NAC')

drop_indices     = listedCompanies_evolution['CountryCode'].isin(world_regions)

selected_regions = listedCompanies_evolution['CountryCode'].isin(selected_countries)



#mortality10_by_region = mortality10_evolution[selected_regions == True]

listedCompanies_by_region = listedCompanies_evolution [drop_indices  == False  ]

listedCompanies_by_region.head()
sns.set_style('darkgrid')

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns_diag = sns.pointplot(x="Year", y="Value", data=listedCompanies_by_region, hue="CountryName", 

                         ax=ax, palette=sns.color_palette("muted"))

sns_diag.set(xlabel='Year', ylabel='Number Domestic companies per Country or Region')

years = list(range(2000,2016))

sns_diag.set_xticklabels(years, rotation=90)

plt.title("Evolution of listed domestic companies per country from 2000 to 2015")

plt.show()
scl = [[0.0, 'rgb(0,245,0)'],   [0.05, 'rgb(0,150, 0)'], [0.1, 'rgb(0,80,80)'],

       [0.15, 'rgb(40,80,80)'], [0.2, 'rgb(200,20,20)'], [1.0, 'rgb(245,0,0)']]

import time

data = dict()

layout = dict()

fig = dict()

for i in range(0,3):

    year_select = listedCompanies_by_region['Year'] == (1975 + i * 10)

    data[str(i)] = [ dict(

        type           ='choropleth',

        colorscale     = scl,

        autocolorscale = False,

        locations      = listedCompanies_by_region[(selected_regions == True)

                                             & (year_select)].CountryCode.values,

        z              = listedCompanies_by_region[(drop_indices == False)

                                             & (year_select)].Value.values,

        zmin           = 0.0,

        zmax           = 444.0,

        text           = listedCompanies_by_region[(selected_regions == True)

                                             & (year_select)].CountryName,

        marker         = dict(

                           line = dict (

                              color = 'rgb(255,255,255)',

                              width = 2

                         ) ),

        colorbar    = dict(

                        title = "Domestic companies per country ")

                      ) 

              ]



    layout[str(i)] = dict(

        title = '{} in {}'.format(listedCompanies_by_region[(drop_indices == False)

                                             & (year_select)].IndicatorName.unique()[0],

                                  listedCompanies_by_region[(drop_indices == False)

                                             & (year_select)].Year.unique()[0]),

        geo = dict(

            scope      = 'World',

            projection = dict( type='Mercator' ),

            showlakes  = True,

            lakecolor  = 'rgb(255, 255, 255)'),

             )  

    fig[str(i)] = dict( data=data[str(i)], layout=layout[str(i)] )

    iplot( fig[str(i)], filename='d3-cloropleth-map' )

    time.sleep(1)