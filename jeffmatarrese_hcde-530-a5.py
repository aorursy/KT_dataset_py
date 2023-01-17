# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # not used



import ast



# plotly libraries

import plotly.express as px

import plotly.graph_objects as go 

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading in dataset and checking to see what it entails

umb = pd.read_csv("../input/seattle-unreinforced-masonry-buildings/unreinforced-masonry-buildings.csv")

umb.head()



# Making it smaller

umb_sm = umb[['Preliminary Risk Category', 'Neighborhood', 'Zip Code', 'No. Stories', 'Retrofit Level', 'Building Use', 'Estimated Number of Occupants']]

umb_sm.head()



# Pulling out location data to try to split it into two columns for mapbox

umb_loc = umb[['Long/Lat']]

loc_list = umb_loc['Long/Lat'].tolist()



# trying to literally evaluate the dictionaries to pull it apart but not working

# for i in loc_list:

#     if i != :

#         ast.literal_eval(i)



# "{'coordinates': [-122.341166, 47.682324], 'type': 'Point'}" 

# dict = {'coordinates': [-122.341166, 47.682324], 'type': 'Point'}

# dict['coordinates'][1] # THIS WORKS, but I just need it to be a dictionary!



# Trying to make a risk pivot table - having a hard time getting it right

umb_risk = umb_sm.groupby(['Preliminary Risk Category', 'Building Use']).count()

umb_risk.head(20)
# reading out the descriptives - commented out to not run on start.

# umb['Neighborhood'].describe()

# umb['Preliminary Risk Category'].describe()

# umb['Building Use'].describe()

# umb['No. Stories'].describe()
hist_risk = px.histogram(umb, x="Preliminary Risk Category", color="Preliminary Risk Category", height=500, width=600)

hist_risk.update_layout(xaxis={'categoryorder':'total descending'}, showlegend=False)

hist_risk.show()



hist_retro = px.histogram(umb, x="Retrofit Level", color="Preliminary Risk Category", height=500, width=700)

hist_retro.update_layout(xaxis={'categoryorder':'total descending'})

hist_retro.show()
hist_hood = px.histogram(umb, x="Neighborhood", color="Preliminary Risk Category", height=700)

hist_hood.update_layout(xaxis={'categoryorder':'total descending'})

hist_hood.update_xaxes(tickangle=45)

hist_hood.show()
hist_type = px.histogram(umb, x="Building Use", color="Preliminary Risk Category")

hist_type.update_layout(xaxis={'categoryorder':'total descending'})

hist_type.update_xaxes(tickangle=45)

hist_type.show()
sunburst = px.sunburst(umb, path=['Preliminary Risk Category','Neighborhood', 'Building Use', 'Estimated Number of Occupants'], maxdepth=2, height=800)

sunburst.show()