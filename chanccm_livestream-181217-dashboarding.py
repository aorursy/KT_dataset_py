# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../input/procurement-notices.csv")
df.columns
df.head()
df['Deadline Date'][:5]
df['Deadline Date'] = pd.to_datetime(df['Deadline Date'])
df['Deadline Date'][:5]
from datetime import datetime, timedelta
df = df[(df['Deadline Date'] >  datetime.now()) | (df['Deadline Date'].notnull())]
df.describe()
# distribution by country

df.groupby(df['Country Code']).count()
number_ID = df['ID'].groupby(df['Country Code']).count()
type(number_ID)
number_ID = number_ID.reset_index()
type(number_ID)
number_ID[:10]
number_ID.plot.bar()
# from https://www.kaggle.com/rohit3463/world-bank-eda-for-dashboard-python-geopandas
import geopandas

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

world.head()
number_country = df['ID'].groupby(df['Country Name']).count()
type(number_country)
number_country = number_country.reset_index()
type(number_country)
number_country.head()
number_country = number_country.set_index('Country Name')
number_country.head()
#world = world.set_index('name').join(number_country)

import pandas as pd

world = pd.merge(world,
                 number_country[['ID']],
                 left_on = 'name',
                 right_on = 'Country Name',
                 how = 'left')
world.head()
#df1.loc[df1['stream'] == 2, 'feat'] = 10
#world['ID'].loc[world['ID'].isna()] = 0.0
world.loc[world['ID'].isna(), 'ID'] = 0.0
#world.head()
world.count()
world.head()
fig, ax = plt.subplots(1, figsize=(20, 12))
world.plot(column='ID',ax = ax)
ax.get_legend()

world.ID.max()
world.ID.min()
# from https://towardsdatascience.com/lets-make-a-map-using-geopandas-pandas-and-matplotlib-to-make-a-chloropleth-map-dddc31c1983d
# set the range for the choropleth
vmin, vmax = 0, 100
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 12))

# create map
world.plot(column='ID', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)
