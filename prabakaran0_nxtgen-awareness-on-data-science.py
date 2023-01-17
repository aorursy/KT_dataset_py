import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly

import plotly.graph_objects as go

import seaborn as sns

from matplotlib.ticker import StrMethodFormatter

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
m = pd.read_csv("../input/map.csv", skipinitialspace = True)
fig = go.Figure(data=go.Choropleth(

    locations = m['CODE'],

    z = m['COUNT'],

    text = m['COUNTRY'],

    colorscale = 'jet',

    autocolorscale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '#',

    colorbar_title = 'Count',

))



fig.update_layout(

    title_text='NxtGen Awareness on Data Science<br>(Click legend to toggle traces)',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    )

)



fig.show()