# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.offline as offline
offline.init_notebook_mode()
import operator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_school = pd.read_csv('../input/2016 School Explorer.csv')
df_school.describe()
categorical = df_school.dtypes[df.dtypes == "object"].index
print(categorical)

df[categorical].describe()
trace1 = go.Histogram(
    x = df_school['School Income Estimate'],
    name = 'School Income Estimate'
)
dat = [trace1]

layout = go.Layout(
    title='School Income Estimate',paper_bgcolor='rgb(243, 243, 243)',plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout = layout)
py.iplot(fig, filename='School-Income-Hist')
from collections import Counter
city_names = []
city_count = []
city_dict = dict(Counter(df_school.City))
city_dict = sorted(city_dict.items(), key=operator.itemgetter(1))
for tup in city_dict:
    city_names.append(tup[0].lower())
    city_count.append(tup[1])

dataa = [go.Bar(
            y= city_names,
            x = city_count,
            width = 0.9,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            )
        )]
layout = go.Layout(
    title='Distribution of Schools ',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=10
    ),
)

fig = go.Figure(data=dataa, layout = layout)
py.iplot(fig, filename='School-City-Bar')

fig2 = {
  "data": [
    {
      "values": city_count,
      "labels": city_names,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    }],
  "layout": {
        "title":"Percentage of Schools in each City",
        "paper_bgcolor":'rgb(243, 243, 243)',"plot_bgcolor":'rgb(178, 255, 102)'
        
    }
}
py.iplot(fig2, filename='School-City-Pie')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,12))
sns.jointplot(x=df_school.Latitude.values, y=df_school.Longitude.values, size=10, color = 'red')
#sns.swarmplot(x="Latitude", y="Longitude", hue="Percent Asian" data=df_school)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
from plotly.offline import init_notebook_mode, iplot
df=df_school

trace0 = go.Box(x=df["School Income Estimate"][df["Community School?"]=="Yes"],name="Community School",boxmean=True)
trace1 = go.Box(x=df["School Income Estimate"][df["Community School?"]=="No"],name="Private School",boxmean=True)
data = [trace0, trace1]
layout = go.Layout(
    title = "Box Plot of estimated income of Community and Private schools",
    margin = go.Margin(l=115)
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)
plt.title('Distribution Of The Average Math Proficiency')
sns.distplot(df_school[df_school['Average Math Proficiency'].notna()]['Average Math Proficiency'])
data = df['Economic Need Index'].dropna()
sns.distplot(data)
def floater(x):
    return float(x.strip('%'))
df["Supportive Environment %"] = df["Supportive Environment %"].astype(str).apply(floater)
df["Supportive Environment %"] = df["Supportive Environment %"].fillna(df["Supportive Environment %"].mean())
trace1 = go.Bar(
    y=df["Supportive Environment Rating"].value_counts(sort=True).index,
    x=df["Supportive Environment Rating"].value_counts(sort=True).values,
    text=df["Supportive Environment Rating"].value_counts(sort=True).values,
    textposition='auto',
    name='Frequency',
    orientation = 'h',
    marker = dict(
        color = 'rgba(62, 66, 75, 0.6)',
        line = dict(
            color = 'rgba(62, 66, 75, 1)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=list(df["Supportive Environment Rating"].unique()),
    x=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].mean() for i in list(df["Supportive Environment Rating"].unique())],
    text=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].mean() for i in list(df["Supportive Environment Rating"].unique())],
    textposition='auto',
    name='Mean',
    orientation = 'h',
    marker = dict(
        color = 'rgba(232, 138, 32, 0.6)',
        line = dict(
            color = 'rgba(232, 138, 32, 1)',
            width = 3)
    )
)

trace3 = go.Bar(
    y=list(df["Supportive Environment Rating"].unique()),
    x=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].median() for i in list(df["Supportive Environment Rating"].unique())],
    text=[df["Supportive Environment %"][df["Supportive Environment Rating"] == i].median() for i in list(df["Supportive Environment Rating"].unique())],
    textposition='auto',
    name='Median',
    orientation = 'h',
    marker = dict(
        color = 'rgba(243, 186, 50, 0.6)',
        line = dict(
            color = 'rgba(243, 186, 50, 1)',
            width = 3)
    )
)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=400, width=800, title='Statistical analysis of Supportive Environment rating',margin=go.Margin(l=100),yaxis=dict(tickangle=45))
iplot(fig, filename='simple-subplot-with-annotations')
sns.countplot(df['Community School?'])
sns.countplot(df['Grade High'])
