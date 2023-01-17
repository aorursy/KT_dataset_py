%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as offline

import plotly.graph_objs as go

offline.init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/SolarSystemAndEarthquakes.csv')

df.head()
df.describe()
trace = go.Scatter(

    x = df["Sun.rectascension"],

    y = df["earthquake.mag"],

    mode='markers'

)



layout = go.Layout(

    title='Correlation between magnitude and Rectascension',

    xaxis=dict(title='Rectascension'),

    yaxis=dict(title='Magnitude'),

    showlegend=True    

)



fig = dict(data=[trace], layout=layout)

offline.iplot(fig)
df = df.sort_values(by='earthquake.time')



trace = go.Scatter(

    x = df["earthquake.time"],

    y = df["earthquake.mag"],

    mode='markers+lines'

)



layout = go.Layout(

    title='Magnitude in these 20years',

    xaxis=dict(title='time'),

    yaxis=dict(title='Magnitude'),

    showlegend=True    

)



fig = dict(data=[trace], layout=layout)

offline.iplot(fig)
trace = go.Histogram(x=df["earthquake.mag"])



layout = go.Layout(

    title='Magnitude Distribution',

    xaxis=dict(title='Magnitude'),

    yaxis=dict(title='Frequency')

)



fig = dict(data=[trace], layout=layout)

offline.iplot(fig)
df["MoonPhase.dynamic"].unique()
dsc = df[df["MoonPhase.dynamic"] == 'dsc']

asc = df[df["MoonPhase.dynamic"] == 'asc']



t1 = go.Box(

    y = dsc["earthquake.mag"].values,

    name = 'Moon Ascending phase',

    marker = dict(

        color = 'rgb(214, 12, 140)'

    )

)



t2 = go.Box(

    y = asc["earthquake.mag"].values,

    name = 'Moon Descending phase',

    marker = dict(

        color = 'rgb(0, 128, 128)'

    )

)



offline.iplot([t1, t2])
t = go.Histogram2d(

    x = df["earthquake.latitude"].values,

    y = df["earthquake.longitude"].values,

    name = 'Position of Earthquake',

    opacity = 0.75    

)



l = go.Layout(

    title = 'Position of Earthquake',

    xaxis = dict(title='Latitude'),

    yaxis = dict(title='Longitude')

)



fig = dict(data=[t], layout=l)

offline.iplot(fig)
import plotly.figure_factory as ff

y = df["earthquake.latitude"].values

x = df["earthquake.longitude"].values



colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]



fig = ff.create_2d_density(

    x, y, colorscale=colorscale,

    hist_color='rgb(0, 128, 222)', point_size=3,

)



offline.iplot(fig)