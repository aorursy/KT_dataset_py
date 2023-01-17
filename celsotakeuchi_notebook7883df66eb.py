# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as pyplot

import plotly.graph_objs as go 

import plotly 

plotly.tools.set_credentials_file(username='celso.takeuchi', api_key='j2SUERJIgo2lAoGMQ0ld')

plotly.tools.set_config_file(world_readable=False,

                             sharing='public')





# Importing data to variables inside Python

# gl_temp = pd.read_csv("../input/GlobalTemperatures.csv")

# gl_temp_City = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv")

gl_temp_Country = pd.read_csv ("../input/GlobalLandTemperaturesByCountry.csv")

# gl_temp_MjCity = pd.read_csv("../input/GlobalLandTemperaturesByMajorCity.csv")

# gl_temp_State = pd.read_csv("../input/GlobalLandTemperaturesByState.csv")



# print(list(gl_temp))

# print(list(gl_temp_City))

# print(list(gl_temp_Country))

# print(list(gl_temp_MjCity))

# print(list(gl_temp_State))

#===========================================================================================#

# Climate informations between 1805 - 1825. 10 years early from Tambora's eruption and 10 year later.

# Compare the climate information of coutries that don't have summer in 1816



# Slicing data

France = gl_temp_Country[gl_temp_Country['Country'] == 'France']

FR = France[(France['dt'] > '1814-01-01') & (France['dt'] < '1816-12-01') ]

England = gl_temp_Country[gl_temp_Country['Country'] == 'United Kingdom']

UK = England[(England['dt'] > '1805-01-01') & (England['dt'] < '1825-12-01') ]




