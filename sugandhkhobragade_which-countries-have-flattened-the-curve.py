

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import geopandas as gpd

from matplotlib import cm

import matplotlib

import plotly.graph_objects as go







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')

df = df[['Date' , 'Name of State / UT' , 'Total Confirmed cases']]

df.columns = ['date', 'state', 'cases']

df = df[['state', 'cases' , 'date']]

df1 = df.pivot_table('cases', ['state'], 'date').reset_index()  #pivoting

#df1.to_csv('covid.csv',index=False)

from IPython.core.display import HTML
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2434552" data-url="https://flo.uri.sh/visualisation/2434552/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')