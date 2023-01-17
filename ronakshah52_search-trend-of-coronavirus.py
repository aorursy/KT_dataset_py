# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/coronaVirus_googleTrends.csv")
fig = go.Figure(data=go.Scatter(x=data["Day"], y=data["Normalized_count"],line=dict(color='firebrick', width=4)))

fig.update_layout(title='Search Trends of CoronaVirus',

                   xaxis_title='Day',

                   yaxis_title='Normalized_count',

                  xaxis = go.layout.XAxis(

        tickangle = 270),

                  template ="plotly_dark")

fig.update_xaxes(nticks=10)

fig.show()