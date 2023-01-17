# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data.head()



# terrorist attacks in South America only

terror_SA = data[(data.region == 3)]



SA_countries = np.asarray(['BRA', 'COL', 'ARG', 'VEN', 'PER', 'CHI', 'ECU', 'BOL', 'PAR', 'URU', 'GUY', 'SUR', 'FAL', 'FRG'])

SA_population = np.asarray([205823665, 47220856, 43886748, 30912302, 30741062, 17650114, 16080778, 10969649, 6862812, 3351016, 735909, 585824, 2931, 231167])



#terrorist attacks per 100,000 people in country

terror_percountrySA = np.asarray(terror_SA.groupby('country').country.count())

terror_percapitaSA = np.round(terror_percountrySA / SA_population * 100000 , 2)



terror_scale = [[0, 'rgb(252, 232, 213)'], [1, 'rgb(240, 140, 45)']]



data = [dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = terror_scale,

        showscale = False,

        locations = SA_countries,

        locationmode = 'South America',

        z = terror_percapitaSA, 

        marker = dict(

            line = dict(

                color = 'rgb(255, 255, 255)',

                width = 2)

            )

        )]



layout = dict(

         title = 'Terrorist Attacks per 100,000 People in South America (1970-2015)',

         geo = dict(

             scope = 'south america',

             projection = dict(type = 'natural earth'),

             countrycolor = 'rgb(255, 255, 255)',

             showlakes = True,

             lakecolor = 'rgb(255, 255, 255)')

         )



figure = dict(data = data, layout = layout)

iplot(figure)
print(terror_percountrySA)