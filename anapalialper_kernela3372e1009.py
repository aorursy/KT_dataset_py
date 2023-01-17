import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image

import os
import seaborn as sns
import plotly.figure_factory as ff
print(os.listdir())
df = pd.read_csv("../input/kepler.csv", error_bad_lines=False)
pd.set_option('display.max_columns', None)

df.head()
df.columns
df.isnull().sum()
def fill_mean_rand(column_name):
    df = pd.read_csv("../input/kepler.csv", error_bad_lines=False)
    mean = df[column_name].mean()

    if (mean*1/2) < (mean*3/2):
        df[column_name] = df[column_name].fillna(pd.Series(np.random.normal(mean*1/20,mean*3/0.1,size=len(df.index))))
    elif mean < 0 :
        df[column_name] = df[column_name].fillna(pd.Series(np.random.normal(mean*3/2,mean*1/2,size=len(df.index))))

    return abs(df[column_name])
def fill_mean_uniform(column_name):
    df = pd.read_csv("../input/kepler.csv", error_bad_lines=False)
    mean = df[column_name].mean()
    
    df[column_name] = df[column_name].fillna(pd.Series(np.random.uniform(mean*1/20,mean*3/0.1,size=len(df.index))))

    return abs(df[column_name])

print(fill_mean_rand("radius"),df["radius"])
plt.figure(figsize=(20,13))
sns.countplot(df["detection_type"])
plt.figure(figsize=(20,13))
sns.countplot(df["discovered"])
trace1 = go.Scatter(
    y = fill_mean_rand('temp_calculated'),
    mode='markers',
    marker=dict(
        size=16,
        color = fill_mean_rand('temp_calculated'),
        colorscale='Reds',
        showscale=True
    )
)

data = [trace1]

py.offline.iplot(data,filename='scatter-plot-with-colorscale')
trace1 = go.Scatter(
    y = df["mass"].dropna(),
    mode='markers',
    marker=dict(
        size=df["radius"].dropna()*40,
        color = fill_mean_rand('temp_calculated'),
        colorscale='Blues',
        showscale=True
    )
)

data = [trace1]

py.iplot(data, filename='scatter-plot-with-colorscale')
df["density"]= df["mass"].dropna()/ (4/3*np.pi*(df["radius"].dropna()**3))
df["density"].dropna()
x = df["density"][0:700].dropna()
hist_data = [x]
group_labels = ['density']

fig = ff.create_distplot(hist_data, group_labels)
py.iplot(fig, filename='Basic Distplot')
trace1 = go.Scatter3d(
    x=fill_mean_rand('star_mass'),
    y=fill_mean_rand('star_distance'),
    z=fill_mean_rand('star_radius'),
    mode='markers',
    marker=dict(
        size=12,
                        # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')
df["luminosity"] = (fill_mean_rand("star_radius")/1)**2 * (fill_mean_rand("star_teff")/5700)**4

trace = go.Scattergl(
    x = df["luminosity"] ,
    y = fill_mean_uniform("star_teff"),
    mode = 'markers',
    marker = dict(
        line = dict(
            width = 1,
            color = '#404040')
    )
)
data = [trace]
py.iplot(data, filename='WebGL100000')

data = [
    go.Scatterpolar(
        r = abs(fill_mean_rand("star_distance"))
,
        theta = df.index ,
        mode = 'markers',
        marker = dict(
            color = 'peru'
        )
    )
]

layout = go.Layout(title='Distance from the star',
    showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = 'polar-basic')