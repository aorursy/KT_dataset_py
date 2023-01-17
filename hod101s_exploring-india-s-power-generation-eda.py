import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
state = pd.read_csv('../input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')

data_df = pd.read_csv('../input/daily-power-generation-in-india-20172020/file.csv') 
state.head()
data_df.head()
plt.show(sns.heatmap(data_df.isnull()))
def findNull(df):

    print("Column\t\t\tNull Percentage\t\t\tNull Records")

    for col in df.columns:

        null_sum = df[col].isnull().sum()

        print(f"{col}\t\t\t{null_sum/len(df)*100}%\t\t\t{null_sum} Null Records")
findNull(data_df)
data_df.fillna(0.0,inplace=True)
data_df["Thermal Generation Actual (in MU)"] = data_df["Thermal Generation Actual (in MU)"].str.replace(',', '').astype(float)

data_df["Thermal Generation Estimated (in MU)"] = data_df["Thermal Generation Estimated (in MU)"].str.replace(',', '').astype(float)
data_df.Date = pd.to_datetime(data_df.Date) 
def boxOut(df, feature1, feature2, title):

    fig = make_subplots(2,1)

    fig.add_trace(go.Box(x=df[feature1], 

                         name="Actual",

                        boxpoints='all',),

                         row=1, col=1)

    fig.add_trace(go.Box(x=df[feature2], 

                         name="Estimated",

                        boxpoints='all',),

                         row=2, col=1)

    fig.update_layout(height=800, 

                      width=800,

                      title=title)

    fig.show()
boxOut(data_df,"Thermal Generation Actual (in MU)","Thermal Generation Estimated (in MU)","Thermal Generation Outliers")
boxOut(data_df,"Nuclear Generation Actual (in MU)","Nuclear Generation Estimated (in MU)","Nuclear Generation Outliers")
boxOut(data_df,"Hydro Generation Actual (in MU)","Hydro Generation Estimated (in MU)","Hydro Generation Outliers")
df = data_df[data_df['Region']=='Northern']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in Northern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in Northern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in Northern Region",legend_orientation="h")
df = data_df[data_df['Region']=='Southern']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in Southern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in Southern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in Southern Region",legend_orientation="h")
df = data_df[data_df['Region']=='Western']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in Western Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in Western Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in Western Region",legend_orientation="h")
df = data_df[data_df['Region']=='Eastern']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in Eastern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in Eastern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in Eastern Region",legend_orientation="h")
df = data_df[data_df['Region']=='NorthEastern']

fig = df.plot(x='Date',y=['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)'])

fig.update_layout(title="Thermal Generation in NorthEastern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)'])

fig.update_layout(title="Nuclear Generation in NorthEastern Region",legend_orientation="h")
fig = df.plot(x='Date',y=['Hydro Generation Actual (in MU)', 'Hydro Generation Estimated (in MU)'])

fig.update_layout(title="Hydro Generation in NorthEastern Region",legend_orientation="h")
def actualVpredicted(df,f1,f2,title):

    fig = go.Figure()

    fig.add_trace(go.Bar(

        x=df.Region,

        y=df.groupby(['Region'])[f1].sum(),

        name='Actual',

        marker_color='indianred'

    ))

    fig.add_trace(go.Bar(

        x=df.Region,

        y=df.groupby(['Region'])[f2].sum(),

        name='Predicted',

        marker_color='lightsalmon'

    ))



    fig.update_layout(barmode='group', title = title)

    fig.show()
actualVpredicted(data_df,"Thermal Generation Actual (in MU)","Thermal Generation Estimated (in MU)","Thermal Generation Outliers")
actualVpredicted(data_df,"Nuclear Generation Actual (in MU)","Nuclear Generation Estimated (in MU)","Nuclear Generation Outliers")
actualVpredicted(data_df,"Hydro Generation Actual (in MU)","Hydro Generation Estimated (in MU)","Hydro Generation Outliers")
data_df['Month'] = data_df.Date.dt.month
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

monthly_distribution(data_df, groupby="Month", dict_features=dict_features, colors=px.colors.qualitative.Prism)

dict_features = {

    "Nuclear Generation Estimated (in MU)": "sum",

    "Nuclear Generation Actual (in MU)": "sum",

}

monthly_distribution(data_df, groupby="Month", dict_features=dict_features, colors=px.colors.qualitative.Antique)

dict_features = {

     "Hydro Generation Estimated (in MU)": "sum",

    "Hydro Generation Actual (in MU)": "sum"

}

monthly_distribution(data_df, groupby="Month", dict_features=dict_features, colors=px.colors.qualitative.Set3)
state.head()
fig = px.bar(state, x='State / Union territory (UT)', y='National Share (%)', color='Area (km2)',height=400)

fig.update_layout(title = 'States Power Generation colored by area')

fig.show()