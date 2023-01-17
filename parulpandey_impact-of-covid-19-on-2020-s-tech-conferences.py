# Import libraries



# import the necessary libraries

import numpy as np 

import pandas as pd 



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pycountry

from plotly.offline import init_notebook_mode, iplot 

import plotly.express as px

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

plt.style.use("fivethirtyeight")# for pretty graphs



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/2020-conferences-cancelled-due-to-coronavirus/2020 Conferences Cancelled Due to Coronavirus - Sheet1.csv',parse_dates=['Scheduled date of physical event'])

df.head()
x = df['Status'].value_counts()

x = pd.DataFrame(x)

x.style.background_gradient(cmap='Reds')
# Cancelled Events



Cancelled = df[df['Status'] == 'Cancelled']

Cancelled.style.set_properties(**{'background-color': '#ffcccb',

                            'color': 'black',

                            'border-color': 'white'})
# Postponed



Postponed = df[df['Status'] == 'Postponed']

Postponed.style.set_properties(**{'background-color': '#fed8b1',

                            'color': 'black',

                            'border-color': 'white'})
# Went Online



Online = df[df['Status'] == 'Online']

Online.style.set_properties(**{'background-color': '#98FB95',

                            'color': 'black',

                            'border-color': 'white'})
# Country where maximum conferences were scheduled

df['Country'].value_counts().to_frame().style.background_gradient(cmap='Reds')

# Areas in US where the conferences were scheduled



US = df[df['Country'] == 'US']

US['Venue'].value_counts().to_frame().style.background_gradient(cmap='Reds')

# Data Source: https://www.vox.com/recode/2020/3/3/21162802/tech-conferences-cancellation-coronavirus

# Includes losses from flights, lodging, food and transportation



loss = {'Mobile World Conference' : 480,

                 'EmTech Asia': 1.7,

                 'SXSW':350,

                 'Game Developers Conference':129.9,

                 'Facebook Global Marketing Summit':19.4,

                 'Adobe Summit':2,

                 'Facebook F8 Conference':12.2,

                 'Shopify Unite': 1.6,

                 'Google I/O': 19.5,

                  'Collision': 49}
economic_loss = pd.DataFrame.from_dict(loss,orient='index')

economic_loss.reset_index(inplace=True)

economic_loss.columns = ['Conference','Loss in Million $']



economic_loss.style.background_gradient(cmap='Reds')