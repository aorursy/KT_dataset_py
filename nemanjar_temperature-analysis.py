# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sb

from matplotlib import pyplot as plt

from ggplot import *

import bokeh as bk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
country_temps = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv', encoding='utf8')



print(country_temps.head())

print('Nrows before dropping NAs: ' + str(country_temps.shape[0]))

country_temps.dropna(subset=['AverageTemperature'], inplace=True)

print('Nrows after dropping NAs: ' + str(country_temps.shape[0]))
country_temps.dt = pd.to_datetime(country_temps.dt, format='%Y-%m-%d')
country_temps['Year'] = country_temps.dt.apply(lambda x: x.year)

country_temps['Month'] = country_temps.dt.apply(lambda x: x.month)

# country_temps.drop('dt', axis=1, inplace=True)
spring_start = pd.to_datetime('21/03/2000', format='%d/%m/%Y').dayofyear

summer_start = pd.to_datetime('21/06/2000', format='%d/%m/%Y').dayofyear

autumn_start = pd.to_datetime('21/09/2000', format='%d/%m/%Y').dayofyear

winter_start = pd.to_datetime('21/12/2000', format='%d/%m/%Y').dayofyear



def get_season(day_of_year):

    

    if day_of_year < spring_start:

        return 'WINTER'

    elif day_of_year < summer_start:

        return 'SPRING'

    elif day_of_year < autumn_start:

        return 'SUMMER'

    elif day_of_year < winter_start:

        return 'AUTUMN'

    else:

        return 'WINTER'

    
country_temps['day_of_year'] = country_temps['dt'].apply(lambda x: x.dayofyear)
country_temps['Season'] = country_temps['day_of_year'].apply(get_season)



print(country_temps.head())
get_season(15)
country_df = country_temps[country_temps.Country == 'Serbia']
p = sb.stripplot(data=country_df, x='Year', y='AverageTemperature', hue='Season');

p.set(title='Serbia')

# Subsample x-ticks

dec_ticks = [y if not x%20 else '' for x,y in enumerate(p.get_xticklabels())]

p.set(xticklabels=dec_ticks);

xticks = p.get_xticklabels()



dec_ticks = [x if not x%10 else '' for x,y in enumerate(xticks)]