import pandas as pd
import numpy as np
import plotly.express as px

dl = pd.read_csv('../input/morphosource-monthly-downloads/dl_number_monthly.csv')
views = pd.read_csv('../input/morphosourcemonthlyviews/view_number_monthly.csv')
fig = px.line(x=list(dl)[:-8], y=dl.iloc[0].values[:-8])
fig.show()
fig = px.line(x=list(dl)[75:-8], y=dl.iloc[0].values[75:-8])
fig.show()
fig = px.line(x=list(views)[:-8], y=views.iloc[0].values[:-8])
fig.show()
fig = px.line(x=list(views)[75:-8], y=views.iloc[0].values[75:-8])
fig.show()