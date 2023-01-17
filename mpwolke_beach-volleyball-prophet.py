#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQlcLz8Yw4CN5GmNHewK2vONrNre9T6cKY0Bmd2zLrfDL2_bN4r&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/beach-volleyball/vb_matches.csv")

df.head()
df.isna().sum()
# filling missing values with NA

#df[['l_p2_tot_hitpct', 'l_p2_tot_aces', 'l_p2_tot_serve_errors', 'l_p2_tot_blocks', 'l_p2_tot_digs']] = df[['l_p2_tot_hitpct', 'l_p2_tot_aces', 'l_p2_tot_serve_errors', 'l_p2_tot_blocks', 'l_p2_tot_digs']].fillna('NA')
fig = px.bar(df, x="l_p2_tot_attacks", y="l_p2_tot_blocks", color="l_p2_tot_attacks",

  animation_frame="year", animation_group="match_num", range_y=[0,4000000000])

fig.show()
px.histogram(df, x='year', color='country')
px.scatter(df, x="l_p2_tot_attacks", y="l_p2_tot_blocks", animation_frame="year", animation_group="match_num",

           size="year", color="tournament", hover_name="circuit",

           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
df["e"] = df["year"]/100

fig = px.scatter(df, x="year", y="l_p2_tot_aces", color="date", error_x="e", error_y="e")

fig.show()
fig = px.bar(df,

             y='l_p2_tot_aces',

             x='year',

             orientation='h',

             color='tournament',

             title='Beach Volleyball',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
from fbprophet import Prophet

df1=df.rename(columns={"date": "ds", "l_p2_tot_aces": "y"})

df1

m = Prophet()

m.fit(df1)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
_ = pd.pivot_table(df, values='l_p2_tot_aces', index='date').plot(style='-o', title="Beach Volleyball")
# 3D Scatter Plot

fig = px.scatter_3d(df, x='l_p2_tot_aces', y='tournament', z='year')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTotnnT6iVKyWCa_JWefiFFV82BRkdEU2iWezEhz3mRHx5i4CUD&usqp=CAU',width=400,height=400)