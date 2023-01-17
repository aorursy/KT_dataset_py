# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/seattle-pet-licenses/Seattle_Pet_Licenses.csv") #import Seattle Pet Licenses data set
totalSpecies = df['Species'].value_counts() #count species by each type

totalSpecies.plot(kind="pie", figsize=(12,12), legend=True, fontsize=0) #plot as a pie chart

totalSpecies
is_dog =  df['Species']=="Dog" #filter data by dogs

df_dogs = df[is_dog] #create new df for dogs

df_dogs["Animal's Name"].value_counts().head(10) #count up all dogs with the same name
df_dogs["Animal's Name"].value_counts().head(10).plot(kind="bar", figsize=(10,10)) #show list as bar chart
is_cat =  df['Species']=="Cat" #filter data by dogs

df_cats = df[is_cat] #create new df for dogs

df_cats["Animal's Name"].value_counts().head(10) #count up all dogs with the same name
df_cats["Animal's Name"].value_counts().head(10).plot(kind="bar", figsize=(10,10)) #show list as bar chart
df_dogs["Primary Breed"].value_counts().head(10) #show top dog breeds in Seattle
df_dogs["Primary Breed"].value_counts().head(10).plot(kind="bar", figsize=(10,10))
df_dogs["ZIP Code"].value_counts().head() #list zip codes with the most dogs registered
#Code to import Plotly API

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_1 = user_secrets.get_secret("plotly")

secret_value_2 = user_secrets.get_secret("plotuser")

import chart_studio

chart_studio.tools.set_credentials_file(username=secret_value_2, api_key=secret_value_1)

import chart_studio.plotly as py

import plotly.graph_objects as go

#Import WA zip code shape file data for map outlines

Zip_Codes = pd.read_csv("../input/washington-zip-code-map-shapes/Zip_Codes.csv")
#Create a map in plotly that centers on Seattle

fig = go.Figure(go.Scattermapbox(

    mode = "markers",

    lon = [-73.605], lat = [45.51],

    marker = {'size': 20, 'color': ["cyan"]}))



fig.update_layout(

    mapbox = {

        'style': "carto-positron",

        'center': { 'lon': -122.335167, 'lat': 	47.608013},

        'zoom': 11, 'layers': [{

            #Sample data from plotly example that needs to be replaced with zip code data

            'source': {

                'type': "FeatureCollection",

                'features': [{

                    'type': "Feature",

                    'geometry': {

                        'type': "MultiPolygon",

                        'coordinates': [[[

                            [-73.606352888, 45.507489991], [-73.606133883, 45.50687600],

                            [-73.605905904, 45.506773980], [-73.603533905, 45.505698946],

                            [-73.602475870, 45.506856969], [-73.600031904, 45.505696003],

                            [-73.599379992, 45.505389066], [-73.599119902, 45.505632008],

                            [-73.598896977, 45.505514039], [-73.598783894, 45.505617001],

                            [-73.591308727, 45.516246185], [-73.591380782, 45.516280145],

                            [-73.596778656, 45.518690062], [-73.602796770, 45.521348046],

                            [-73.612239983, 45.525564037], [-73.612422919, 45.525642061],

                            [-73.617229085, 45.527751983], [-73.617279234, 45.527774160],

                            [-73.617304713, 45.527741334], [-73.617492052, 45.527498362],

                            [-73.617533258, 45.527512253], [-73.618074188, 45.526759105],

                            [-73.618271651, 45.526500673], [-73.618446320, 45.526287943],

                            [-73.618968507, 45.525698560], [-73.619388002, 45.525216750],

                            [-73.619532966, 45.525064183], [-73.619686662, 45.524889290],

                            [-73.619787038, 45.524770086], [-73.619925742, 45.524584939],

                            [-73.619954486, 45.524557690], [-73.620122362, 45.524377961],

                            [-73.620201713, 45.524298907], [-73.620775593, 45.523650879]

                        ]]]

                    }

                }]

            },

            'type': "fill", 'below': "traces", 'color': "royalblue"}]},

    margin = {'l':0, 'r':0, 'b':0, 't':0})



fig.show()