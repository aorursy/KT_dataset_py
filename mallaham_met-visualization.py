import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/MetObjects.csv')
data.info()
sCountry = pd.DataFrame(data['Country'].value_counts())

sCountry.columns = ['Count']

sCountry['Country'] = sCountry.index.tolist()

sCountry.sort_values(by="Count",ascending=False)

sCountry = sCountry.reset_index(drop=True)

sCountry
#The following two lines are important to use plotly offline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
plt.figure(figsize=(80,80))

temp = sCountry[sCountry['Count']>=100]

init_notebook_mode(connected=True)

labels=temp['Country']

values=temp['Count']

trace=go.Pie(labels=labels,values=values)



iplot([trace])
data.keys()
sArtist = pd.DataFrame(data['Artist Display Name'].value_counts())

sArtist.columns=['Count']

sArtist['Name'] = sArtist.index.tolist()

sArtist.sort_values(by="Count",ascending=False)

sArtist = sArtist.reset_index(drop=True)

sArtist.head(5)
plt.figure(figsize=(80,80))

temp = sArtist[sArtist['Count']>=1000]

init_notebook_mode(connected=True)

labels=temp['Name']

values=temp['Count']

trace=go.Pie(labels=labels,values=values)



iplot([trace])
sDpt = pd.DataFrame(data['Department'].value_counts())

sDpt.columns=['Count']

sDpt['Name'] = sDpt.index.tolist()

sDpt.sort_values(by="Count",ascending=False)

sDpt = sDpt.reset_index(drop=True)

sDpt.head(5)
plt.figure(figsize=(80,80))

temp = sDpt[sDpt['Count']>=1000]

init_notebook_mode(connected=True)

labels=temp['Name']

values=temp['Count']

trace=go.Pie(labels=labels,values=values)



iplot([trace])