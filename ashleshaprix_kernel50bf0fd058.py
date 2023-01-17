import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go





rain_df = pd.read_csv("../input/chennai_reservoir_rainfall.csv")

rain_df["Date"] = pd.to_datetime(rain_df["Date"], format='%d-%m-%Y')



rain_df["total"] = rain_df["POONDI"] + rain_df["CHOLAVARAM"] + rain_df["REDHILLS"] + rain_df["CHEMBARAMBAKKAM"]

rain_df["total"] = rain_df["POONDI"] + rain_df["CHOLAVARAM"] + rain_df["REDHILLS"] + rain_df["CHEMBARAMBAKKAM"]



def bar_plot(cnt_srs, color):

    trace = go.Bar(

        x=cnt_srs.index[::-1],

        y=cnt_srs.values[::-1],

        showlegend=False,

        marker=dict(

            color=color,

        ),

    )

    return trace



rain_df["YearMonth"] = pd.to_datetime(rain_df["Date"].dt.year.astype(str) + rain_df["Date"].dt.month.astype(str), format='%Y%m')



cnt_srs = rain_df.groupby("YearMonth")["total"].sum()

trace5 = bar_plot(cnt_srs, 'red')



fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                          subplot_titles=["Total rainfall in all four reservoir regions - in mm"])

fig.append_trace(trace5, 1, 1)





fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
rain_df["Year"] = pd.to_datetime(rain_df["Date"].dt.year.astype(str), format='%Y')



cnt_srs = rain_df.groupby("Year")["total"].sum()

trace5 = bar_plot(cnt_srs, 'red')



fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                          subplot_titles=["Total yearly rainfall in all four reservoir regions - in mm"])

fig.append_trace(trace5, 1, 1)





fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')


