# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# import graph objects as "go"

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
import pandas as pd

DEMO_DS_29112019163028002 = pd.read_csv("../input/complete-educational-stats-unesco/EDULIT_DS_30112019025843883.csv")
general = DEMO_DS_29112019163028002[DEMO_DS_29112019163028002['Country'] == 'Brazil']

BrazilCoef = general[general['Indicator'] == 'Enrolment in pre-primary education, both sexes (number)']



general = DEMO_DS_29112019163028002[DEMO_DS_29112019163028002['Country'] == 'Argentina']

ArgCoef = general[general['Indicator'] == 'Enrolment in pre-primary education, both sexes (number)']



general = DEMO_DS_29112019163028002[DEMO_DS_29112019163028002['Country'] == 'Chile']

ChileCoef = general[general['Indicator'] == 'Enrolment in pre-primary education, both sexes (number)']



trace1 = go.Scatter(

                x = ArgCoef['Time'],

                y = ArgCoef['Value'],

                name = "Argentina",

                marker=dict(

                        size=9,

                        color = ('aqua'))

)

trace2 = go.Scatter(

                x = ChileCoef['Time'],

                y = ChileCoef['Value'],

                name = "Chile",

                marker=dict(

                    size=9,

                    color = ('navy'))

    )

trace3 = go.Scatter(

                x = BrazilCoef['Time'],

                y = BrazilCoef['Value'],

                name = "Brazil",

                marker=dict(

                    size=9,

                    color = ('red'))

    )

data = [trace1, trace2, trace3]

layout = dict(title = 'Enrolment in pre-primary education, both sexes',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'Enrolment in pre-primary education, both sexes')

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)