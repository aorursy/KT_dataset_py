# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as df

import plotly.graph_objs as go

import plotly.offline as po

from plotly.subplots import make_subplots
data = df.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")

maxdate = data.date.max()

datadt = data.loc[data.date == maxdate]
cases = datadt.pivot_table(index="state", values="cases", aggfunc="sum")

death = datadt.pivot_table(index="state", values="deaths", aggfunc="sum")
subfig = make_subplots(specs=[[{"secondary_y": True}]])

subfig.add_trace(

    go.Bar(x=cases.index, y=cases.values.flatten(), name="confirmed data"),

    secondary_y=False,

)

subfig.add_trace(

    go.Scatter(x=death.index, y=death.values.flatten(), name="deaths data"),

    secondary_y=True,

)



# Add figure title

subfig.update_layout(

    title_text="COVID-19 Confirmes/Deaths"

)



# Set x-axis title

subfig.update_xaxes(title_text="States")



# Set y-axes titles

subfig.update_yaxes(title_text="<b>Confirmed</b> Cases", secondary_y=False)

subfig.update_yaxes(title_text="No of <b>Deaths</b>", secondary_y=True)



subfig.show()