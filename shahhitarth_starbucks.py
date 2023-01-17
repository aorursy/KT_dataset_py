import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install pyforest
from pyforest import *

import plotly

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()

import os

import warnings

warnings.filterwarnings('ignore')

import mpl_toolkits

from IPython.display import Image

Image(url='https://rde-stanford-edu.s3.amazonaws.com/Hospitality/Images/starbucks-header.jpg')
df = pd.read_csv('../input/directory.csv')

df.head()
df.info()
def update_column(column):

    return column.replace(' ', '_').lower()
starbucks = df.copy()
starbucks.columns = starbucks.columns.map(update_column)
starbucks.info()
starbucks.isnull().sum()
starbucks.ownership_type.unique()
starbucks.info()
starbucks.country.unique()
country_indices, country_labels = starbucks.country.factorize()

country_labels
country_indices
starbucks['country_indice'] = country_indices
starbucks['country'].value_counts()
sns.set(style='dark', context='talk')

sns.countplot(x='ownership_type', data=starbucks, palette='BuGn_d')
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

ax.set(title = 'Top 10 Countries with most number of Starbucks outlets')

starbucks.country.value_counts().head(10).plot(kind='bar', color='blue')
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

ax.set(title='Top 10 Store Names with most Outlets')

starbucks.store_name.value_counts().head(10).plot(kind='bar', color='orange')
starbucks['state/province'].value_counts()
usa_states = starbucks[starbucks['country'] == 'US']

usa_states['state/province'].value_counts().head(1)
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

ax.set(title='Top 10 States of USA with most outlets')

usa_states['state/province'].value_counts().head(10).plot(kind='bar', color='purple')
starbucks.brand.value_counts()
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

ax.set(title='Brand under which Starbucks operate')

starbucks.brand.value_counts().head(10).plot(kind='bar', color='pink')
import os

os.environ['PROJ_LIB'] = 'C:\\Users\\HITARTH SHAH\\Anaconda3\\ANA_NAV\\Library\\share'

from mpl_toolkits.basemap import Basemap
!pip install --upgrade pip
!conda install -c conda-forge basemap-data-hires --yes
plt.figure(figsize=(12,9))

m=Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='h')

m.drawcoastlines()

m.drawcountries()

m.drawmapboundary()

x, y=m(list(starbucks['longitude'].astype(float)), list(starbucks['latitude'].astype(float)))

m.plot(x,y,'bo', markersize=5, alpha=0.6, color='blue')

plt.title('Starbucks Stores Across the World')

plt.show()
plt.figure(figsize=(12,9))

m=Basemap(projection='mill', llcrnrlat=20, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-60, resolution='h')

m.drawcoastlines()

m.drawcountries()

m.drawmapboundary()

x, y=m(list(starbucks['longitude'].astype(float)), list(starbucks['latitude'].astype(float)))

m.plot(x,y,'bo', markersize=5, alpha=0.6, color='blue')

plt.title('Extinct and Endangered Languages in USA')

plt.show()