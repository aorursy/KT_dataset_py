import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

import plotly.express as px

from plotnine import *

import folium
from IPython.display import Javascript

from IPython.core.display import display, HTML



cdr = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

idr = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue

s = '#f0134d'

h = '#12cad6'

e = '#4a47a3'

m = '#42e6a4'

c = '#333333'



shemc = [s, h, e, m, c]

sec = [s, e, c]
from google.colab import  drive

drive.mount('/content/drive')
corno19=pd.read_csv("/content/drive/My Drive/corno19.csv",header=None)
corno19.head(2)
corono=corno19.rename(columns=corno19.iloc[1]).drop(corno19.index[1])
corono.head(8)
corono1=corono.drop(corono.index[[0]])
corono1.head(8)
df = pd.DataFrame(corono1)
df.head(2)
corono2 = df.dropna(axis=1, how="all")
corono2.head(2)
coron3=corono2.reset_index(drop=True, inplace=True)
corono2.head(2)
corono3=corono2.drop(columns=1,axis=1)
corono3.head(3)
corono4=corono3.drop(corono3.index[[1]])

corono4=corono4.drop(corono3.index[[0]])

corono4.head(3)
fig = px.bar(corono4.sort_values('Deaths', ascending=False),

             x="State", y="Deaths", color='Deaths', 

             text='State', orientation='v', title='covid-19', 

             range_x=[-5,50],

             color_discrete_sequence = [h, c, s, m, e])



fig.update_traces(textposition='auto')

fig.show()
fig = px.bar(corono4.sort_values('Active', ascending=False),

             x="State", y="Active", color='Active', 

             text='State', orientation='v', title='covid-19', 

             range_x=[-5,50],

             color_discrete_sequence = [h, c, s, m, e])



fig.update_traces(textposition='inside')



fig.show()
fig = px.line(corono4, x="Last_Updated_Time", y="Active", color='Active', 

              title="Worldwide Confirmed/Death Cases Over Time")

fig.show()