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
emmysDf = pd.read_csv("/kaggle/input/emmy-awards/the_emmy_awards.csv")

print("Emmys dataset has %d rows and %d columns" %(emmysDf.shape[0], emmysDf.shape[1]))

emmysDf.head(n=10)
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

from plotly.offline import iplot
trace2 = go.Bar(x = emmysDf[emmysDf['win'] == True]['category'].groupby([emmysDf['year']]).agg('count').sort_values(ascending = False).index,

                y = emmysDf[emmysDf['win'] == True]['category'].groupby([emmysDf['year']]).agg('count').sort_values(ascending = False).values,

                name = "wins",

                marker=dict(line=dict(width=0.5), color='#779ecb'), 

                text = emmysDf[emmysDf['win'] == True]['category'].groupby([emmysDf['year']]).agg('count').sort_values(ascending = False).index)

data = [trace2]

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title="Number of categories over the years",

    font=dict(

        size=10,

        color="#000000"

    )

)

iplot(fig)
trace1 = go.Bar(x = emmysDf['company'].value_counts()[:10].index,

                y = emmysDf['company'].value_counts()[:10].values,

                name = "nominations",

                text = emmysDf['company'].value_counts()[:10].index)



trace2 = go.Bar(x = emmysDf[emmysDf["win"] == True]['company'].value_counts()[:10].index,

                y = emmysDf[emmysDf["win"] == True]['company'].value_counts()[:10].values,

                name = "wins",

                marker=dict(line=dict(width=0.5), color='green'), 

                text = emmysDf[emmysDf["win"] == True]['company'].value_counts()[:10].index)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title="Nominations and wins by top 10 studios",

    font=dict(

        size=10,

        color="#000000"

    )

)

iplot(fig)
trace2 = go.Bar(x = emmysDf['nominee'].value_counts()[:15].index,

                y = emmysDf['nominee'].value_counts()[:15].values,

                name = "wins",

                marker=dict(line=dict(width=0.5), color='blue'), 

                text = emmysDf['nominee'].value_counts()[:15].index)

data = [trace2]

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title="Most nominations by shows over the years",

    font=dict(

        size=10,

        color="#000000"

    )

)

iplot(fig)
trace2 = go.Bar(x = emmysDf[emmysDf.year >= 2000]['nominee'].value_counts()[:15].index,

                y = emmysDf[emmysDf.year >= 2000]['nominee'].value_counts()[:15].values,

                name = "wins",

                marker=dict(line=dict(width=0.5), color='blue'), 

                text = emmysDf[emmysDf.year >= 2000]['nominee'].value_counts()[:15].index)

data = [trace2]

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title="Most nominations by shows in past 10 years",

    font=dict(

        size=10,

        color="#000000"

    )

)

iplot(fig)
trace2 = go.Bar(x = emmysDf[emmysDf.win == True]['nominee'].value_counts()[:15].index,

                y = emmysDf[emmysDf.win == True]['nominee'].value_counts()[:15].values,

                name = "wins",

                marker=dict(line=dict(width=0.5), color='green'), 

                text = emmysDf[emmysDf.win == True]['nominee'].value_counts()[:15].index)

data = [trace2]

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title="Most wins by nominees over the years",

    font=dict(

        size=10,

        color="#000000"

    )

)

iplot(fig)
trace2 = go.Bar(x = emmysDf[emmysDf.year >= 2000][emmysDf.win == True]['nominee'].value_counts()[:15].index,

                y = emmysDf[emmysDf.year >= 2000][emmysDf.win == True]['nominee'].value_counts()[:15].values,

                name = "wins",

                marker=dict(line=dict(width=0.5), color='green'), 

                text = emmysDf[emmysDf.year >= 2000][emmysDf.win == True]['nominee'].value_counts()[:15].index)

data = [trace2]

layout = go.Layout(barmode = "group")



fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title="Most wins in past 10 years",

    font=dict(

        size=10,

        color="#000000"

    )

)

iplot(fig)
print("Most common categories")

emmysDf['year'].groupby([emmysDf['category']]).agg('count').sort_values(ascending = False)[:10]
print("Least common categories")

emmysDf['year'].groupby([emmysDf['category']]).agg('count').sort_values(ascending = True)[:20]