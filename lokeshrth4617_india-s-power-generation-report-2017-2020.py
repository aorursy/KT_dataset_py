import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.io as pio

import pandas_profiling

import os

import calendar

pio.templates.default = "plotly_dark"

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data1 = pd.read_csv('../input/daily-power-generation-in-india-20172020/file_02.csv')

data1.head()
data1['Date'] = pd.to_datetime(data1['Date'])
data1['Thermal Generation Estimated (in MU)'] = data1['Thermal Generation Estimated (in MU)'].str.replace(',','').astype('float')

data1['Thermal Generation Estimated (in MU)'].values
data1['Thermal Generation Actual (in MU)'] = data1['Thermal Generation Actual (in MU)'].str.replace(',','').astype('float')

data1['Thermal Generation Actual (in MU)'].values
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

time_series_overall(data1, groupby="Date", dict_features=dict_features)



dict_features = {

    "Nuclear Generation Estimated (in MU)": "sum",

    "Nuclear Generation Actual (in MU)": "sum",

}

time_series_overall(data1, groupby="Date", dict_features=dict_features)



dict_features = {

     "Hydro Generation Estimated (in MU)": "sum",

    "Hydro Generation Actual (in MU)": "sum"

}

time_series_overall(data1, groupby="Date", dict_features=dict_features)
state_df = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')

state_df.head()
state = state_df.groupby('Region')['National Share (%)'].sum().sort_values(ascending = False)

state.index

fig = px.pie(state,values = state.values, names=state.index,

            title='Distribution of Power Region Wise')

fig.show()
fig = px.bar(state_df.nlargest(15, "National Share (%)"),

       x = 'National Share (%)',

       y = 'State / Union territory (UT)',

       text="National Share (%)",

      color ='State / Union territory (UT)')

fig.show()