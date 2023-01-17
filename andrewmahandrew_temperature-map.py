# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt  # To visualize

from sklearn import metrics

from datetime import date

from dateutil.rrule import rrule, DAILY

import statsmodels.formula.api as smf

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std

from patsy import dmatrices

import json

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

    

    break 



# Any results you write to the current directory are saved as output.

#load the temperature data by country

temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')

temperatures = temperatures.groupby(['Country']).max().reset_index()



#load the temperature data by state

temperaturesStates = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv')

temperaturesStates = temperaturesStates.groupby(['State']).max().reset_index()







print(temperatures)

print(temperaturesStates)





temperatures
#https://towardsdatascience.com/why-and-how-to-use-merge-with-pandas-in-python-548600f7e738

#Original temperature data doesnt have country codes which was neccesary for cloropleth maps

#Country codes are manually added by using country-code/country_code.csv data set



countryCodes = pd.read_csv('/kaggle/input/country-code/country_code.csv')

countryCodes.rename(columns = {'Country_name':'Country'}, inplace = True) 

#temperatures = pd.concat([temperatures, countryCodes], axis=1, sort=False)

ccTemperature = pd.merge(temperatures, countryCodes, how='outer',

                  left_on='Country', right_on='Country')



ccTemperature = ccTemperature.dropna(how='any',axis=0)





ccTemperature





# Same thing with states

stateCodes = pd.read_csv('/kaggle/input/us-state-county-name-codes/states.csv')



csTemperature = pd.merge(temperaturesStates, stateCodes, how='outer',

                  left_on='State', right_on='State')

csTemperature = csTemperature.dropna(how='any',axis=0)



csTemperature
#Cloropleth map: https://plotly.com/python/choropleth-maps/

tempFig = go.Figure(data=go.Choropleth(

    locations = ccTemperature['code_3digit'],

    z = temperatures['AverageTemperature'],

    text = temperatures['Country'],

    colorscale = 'YLORRD',                                        #https://blogs.sas.com/content/iml/2014/10/01/colors-for-heat-maps.html#prettyPhoto/0/

    autocolorscale = False,                                       

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_ticksuffix = ' degrees Celcius',

    colorbar_title = 'Temperature around the world',

))



tempFig.update_layout(

    title_text='Temperature around the world',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),

    annotations = [dict(

        x=0.55,

        y=0.1,

        xref='paper',

        yref='paper',

        showarrow = False

    )]

)



tempFig.show()
#https://towardsdatascience.com/why-and-how-to-use-merge-with-pandas-in-python-548600f7e738

tempFigStates = go.Figure(data=go.Choropleth(

    locations = csTemperature['Abbreviation'],

    z = temperaturesStates['AverageTemperature'],

    text = temperaturesStates['Country'],

    colorscale = 'YLORRD',                           #https://blogs.sas.com/content/iml/2014/10/01/colors-for-heat-maps.html#prettyPhoto/0/

    locationmode='USA-states',                       # uses the US state codes instead of country codes

    autocolorscale = False,

    reversescale = False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_ticksuffix = ' degrees Celcius',

    colorbar_title = 'Temperature in the states',

))



tempFigStates.update_layout(

    title_text='Temperature around the world',

    geo=dict(scope='usa',                            # Focus on just the united states

        showframe=False,

        showcoastlines=False,

    ),

    annotations = [dict(

        x=0.55,

        y=0.1,

        xref='paper',

        yref='paper',

        showarrow = False

    )]

)



tempFigStates.show()