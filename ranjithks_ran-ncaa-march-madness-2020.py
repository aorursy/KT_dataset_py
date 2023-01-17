# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sb

import plotly.express as px



%matplotlib inline
dfTeams = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')

dfGameCities = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MGameCities.csv')

dfCities = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/Cities.csv')

dfSeasons = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WSeasons.csv')

dfNTDR = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

dfTourneyGames = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MConferenceTourneyGames.csv')

dfTourneySeeds = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneySeeds.csv')
dfSeasons
dfTourneySeeds
dfTourneyGames
#dfTeams.style.background_gradient(cmap='Pastel1_r')

dfTeams
dfGameCities
dfCities
dfGameCities.CityID.unique().shape
dfCities.CityID.unique().shape
import plotly.graph_objects as go

df_plot = dfGameCities.merge(dfCities, left_on='CityID', right_on='CityID')



fig = go.Figure(data=go.Choropleth(

    locations=df_plot['State'], # Spatial coordinates

    z = df_plot['CityID'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'viridis'

))



fig.update_layout(

    title_text = 'NCAA Game Cities',

    geo_scope='usa', # limite map scope to USA

)
#dfNTDR.style.background_gradient(cmap='Pastel1_r')

dfNTDR
dfNTDR.shape
df_plot