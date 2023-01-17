import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.io as pio

import pandas_profiling

import os

import calendar
PATHS = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        PATHS.append(os.path.join(dirname, filename))

PATHS.sort()
df = pd.read_csv(PATHS[-1], thousands=',')

pio.templates.default = "plotly_dark"
df['Year']=[d.split('-')[0] for d in df.Date]

df['Month']=[d.split('-')[1] for d in df.Date]

df['Day']=[d.split('-')[2] for d in df.Date]
def time_series_overall(df, groupby, dict_features, filter=None):

    temp = df.groupby(groupby).agg(dict_features)

    fig = go.Figure()

    for f,c in zip(dict_features, px.colors.qualitative.D3):

        fig.add_traces(go.Scatter(y=temp[f].values,

                              x=temp.index,

                              name=f,

                              marker=dict(color=c)

                             ))

    fig.update_traces(marker_line_color='rgb(255,255,255)',

                      marker_line_width=2.5, opacity=0.7)

    fig.update_layout(

                    width=1000,

                    xaxis=dict(title="Date", showgrid=False),

                    yaxis=dict(title="MU", showgrid=False),

                    legend=dict(

                                x=0,

                                y=1.2))

                                

    fig.show()
dict_features = {

    "Thermal Generation Estimated (in MU)": "sum",

    "Thermal Generation Actual (in MU)": "sum",

   

}

time_series_overall(df, groupby="Date", dict_features=dict_features)

dict_features = {

    "Nuclear Generation Estimated (in MU)": "sum",

    "Nuclear Generation Actual (in MU)": "sum",

}

time_series_overall(df, groupby="Date", dict_features=dict_features)

dict_features = {

     "Hydro Generation Estimated (in MU)": "sum",

    "Hydro Generation Actual (in MU)": "sum"

}

time_series_overall(df, groupby="Date", dict_features=dict_features)

def power_gen_by_region(df, f_1, f_2):

    fig = make_subplots(2,1)

    temp = df.groupby("Region").agg({f_1: "sum",

                                     f_2: "sum"}).reset_index()

    fig.add_trace(go.Box(x=df[f_1], 

                         orientation='h', 

                         name="Actual",

                         marker=dict(color=px.colors.qualitative.Pastel[1])), 

                         row=1, col=1)

    fig.add_trace(go.Box(x=df[f_2], 

                         orientation='h', 

                         name="Estimated",

                         marker=dict(color=px.colors.qualitative.Pastel[2])), 

                         row=1, col=1)

    fig.add_trace(go.Bar(y=temp["Region"],

                         x=temp[f_1],

                         orientation='h',

                         text=temp[f_1],

                         textposition="inside",

                         texttemplate='%{text:.2s}',

                         name="Actual",

                         marker=dict(color=px.colors.qualitative.Pastel[1])), row=2, col=1)

    fig.add_trace(go.Bar(y=temp["Region"],

                         x=temp[f_2],

                         text=temp[f_2],

                         textposition="inside",

                         texttemplate='%{text:.2s}',

                         orientation='h',

                         name="Estimated",

                         marker=dict(color=px.colors.qualitative.Pastel[2])), row=2, col=1)

    fig.update_traces(opacity=0.7,

                    )

    fig.update_traces(opacity=0.7)

    fig.update_layout(height=800, 

                      width=800,

                      xaxis=dict(showgrid=False),

                      yaxis=dict(showgrid=False),

                      title=" ".join(s for s in f_1.split()[:2]) + " by region")

    fig.show()

    
for f_1,f_2 in zip(df.columns.tolist()[2:8:2], df.columns.tolist()[3:8:2]):

    power_gen_by_region(df, f_1, f_2)
df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[int(x)])
def monthly_distribution(df, groupby, dict_features, colors, filter=None):

    temp = df.groupby(groupby).agg(dict_features)

    fig = go.Figure()

    for f,c in zip(dict_features, colors):

        fig.add_traces(go.Bar(y=temp[f].values,

                              x=temp.index,

                              name=f,

                              text=temp[f].values,

                              marker=dict(color=c)

                             ))

    fig.update_traces(marker_line_color='rgb(255,255,255)',

                      marker_line_width=2.5,

                      opacity=0.7,

                      textposition="outside",

                      texttemplate='%{text:.2s}')

    fig.update_layout(

                    width=1000,

                    xaxis=dict(title="Month", showgrid=False),

                    yaxis=dict(title="MU", showgrid=False),

                    legend=dict(

                                x=0,

                                y=1.2))

                                

    fig.show()
dict_features = {

    "Thermal Generation Estimated (in MU)": "sum",

    "Thermal Generation Actual (in MU)": "sum",

   

}

monthly_distribution(df, groupby="Month", dict_features=dict_features, colors=px.colors.qualitative.Prism)

dict_features = {

    "Nuclear Generation Estimated (in MU)": "sum",

    "Nuclear Generation Actual (in MU)": "sum",

}

monthly_distribution(df, groupby="Month", dict_features=dict_features, colors=px.colors.qualitative.Antique)

dict_features = {

     "Hydro Generation Estimated (in MU)": "sum",

    "Hydro Generation Actual (in MU)": "sum"

}

monthly_distribution(df, groupby="Month", dict_features=dict_features, colors=px.colors.qualitative.Set3)
state_df = pd.read_csv(PATHS[0])
fig = px.bar(state_df.nlargest(10, "National Share (%)"),

             y="State / Union territory (UT)",

             x="National Share (%)",

             color="State / Union territory (UT)",

             text="National Share (%)",

             color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(255,255,255)',

                  marker_line_width=2.5,

                  textposition="outside",

                  texttemplate='%{text:.2s%}',

                  )

fig.update_layout(width=800,

                  title="Top 10 States by power share",

                    yaxis=dict(showgrid=False))

fig.show()
fig = px.bar(state_df.nlargest(10, "National Share (%)"),

             y="State / Union territory (UT)",

             x="Area (km2)",

             color="State / Union territory (UT)",

             text="Area (km2)",

             color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(255,255,255)',

                  marker_line_width=2.5,

                  textposition="inside",

                  texttemplate='%{text:.2s}',

                  )

fig.update_layout(width=800,

                  title="Top 10 States by area",

                    yaxis=dict(showgrid=False,showticklabels=False))

fig.show()