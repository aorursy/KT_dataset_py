import pandas as pd

import numpy as np



import plotly.graph_objs as go

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt



from plotly.offline import init_notebook_mode, iplot

import plotly.offline as py

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')
AppsInfo = pd.read_csv("../input/top-indian-educational-apps-reviews/app_info.csv")

AppsInfo.head(11)
AppsInfo.shape
AppsInfo.info()
AppsInfo.select_dtypes(include = ['object']).columns.values
AppsInfo.select_dtypes(include = ['int64', 'float64']).columns.values
AppsInfo.genreId.unique()
AppsInfo.androidVersion.unique()
fig = px.bar(AppsInfo, x="androidVersion", y="size", color='name', orientation='h')

py.iplot(fig, filename='bar')
fig = px.scatter(AppsInfo, y='ratings', x='minInstalls', color='name', height=600, width=500)  

fig.update_traces(marker=dict(size=12, line=dict(width=2,color='DarkSlateGrey')), selector=dict(mode='markers'))

#fig.show()

py.iplot(fig, filename='scatter')