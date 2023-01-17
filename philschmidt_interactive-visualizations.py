import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output



%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



print(check_output(["ls", "../input"]).decode("utf8"))



macro = pd.read_csv('../input/macro.csv')

df = pd.read_csv("../input/train.csv")

df.head() 
print(list(df.columns))
macro.head()
df.info()
plt.figure(figsize=(12,8))

sns.distplot(df['price_doc'])
boxes = []

for cafe_count in df['cafe_count_500'].unique():

    y = df[df['cafe_count_500'] == cafe_count]['price_doc'].values

    b = go.Box(

        x=y,

        name = 'Cafe count {}'.format(cafe_count),

    )

    boxes.append(b)



py.iplot(boxes)
boxes = []

for floor in df['floor'].unique():

    y = df[df['floor'] == floor]['price_doc'].values

    b = go.Box(

        x=y,

        name = 'Floor {}'.format(floor),

    )

    boxes.append(b)



py.iplot(boxes)
boxes = []

for num_room in df['num_room'].unique():

    y = df[df['num_room'] == num_room]['price_doc'].values

    b = go.Box(

        x=y,

        name = '#Rooms {}'.format(num_room),

    )

    boxes.append(b)



py.iplot(boxes)
#boxes = []

#for build_year in df['build_year'].unique():

#    y = df[df['build_year'] == build_year]['price_doc'].values

#    b = go.Box(

#        x=y,

#        name = 'Buildyear {}'.format(build_year),

#    )

#    boxes.append(b)

#py.iplot(boxes)
# water_km
df = df[df['full_sq'] < 300]

df = df[df['life_sq'] < 300]

sns.jointplot("full_sq", "life_sq", data=df.sample(10000), kind="reg")
sns.jointplot("park_km", "preschool_km", data=df.sample(10000), kind="reg")
sns.jointplot("park_km", "full_sq", data=df.sample(10000), kind="reg")
df['ts'] = pd.to_datetime(df['timestamp'])

df.ts.head()
df['rolling_price_300'] = df['price_doc'].rolling(window=300, center=False).mean()

df['rolling_price_1200'] = df['price_doc'].rolling(window=1200, center=False).mean()

ax = df.sort_values('ts').plot(x='ts', y='rolling_price_300', figsize=(12,8))

df.sort_values('ts').plot(x='ts', y='rolling_price_1200', color='r', ax=ax)

plt.ylabel('price')
df = df.sort_values('water_km')

df['rolling_price_300_water'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='water_km', y='rolling_price_300_water', figsize=(12,8))
df = df.sort_values('park_km')

df['rolling_price_300_park_km'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='park_km', y='rolling_price_300_park_km', figsize=(12,8))
df = df.sort_values('big_church_km')

df['rolling_price_300_big_church_km'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='big_church_km', y='rolling_price_300_big_church_km', figsize=(12,8))
df = df.sort_values('mosque_km')

df['rolling_price_300_mosque_km'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='mosque_km', y='rolling_price_300_mosque_km', figsize=(12,8))
df = df.sort_values('preschool_km')

df['rolling_price_300_preschool_km'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='preschool_km', y='rolling_price_300_preschool_km', figsize=(12,8))
df = df.sort_values('full_sq')

df['rolling_price_300_full_sq'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='full_sq', y='rolling_price_300_full_sq', figsize=(12,8))
df = df.sort_values('area_m')

df['rolling_price_300_area_m'] = df['price_doc'].rolling(window=300, center=False).mean()

ax = df.plot(x='area_m', y='rolling_price_300_area_m', figsize=(12,8))
from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(df.fillna(0.0)[['area_m', 'full_sq', 'num_room']])

X = df.fillna(0.0)[['area_m', 'full_sq', 'num_room']].values

X.shape
df['log_price'] = np.log10(df['price_doc'].values)
trace1 = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=X[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = df['log_price'].values,

        colorscale = 'Portland',

        colorbar = dict(title = 'price_doc'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.5

    )

)



data=[trace1]

layout=dict(

    height=800,

    width=800,

    scene=go.Scene(

        xaxis=go.XAxis(title='area_m'),

        yaxis=go.YAxis(title='full_sq'),

        zaxis=go.ZAxis(title='num_room')

    ),

    title='Prices by three dimensions'

)

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
X = df.fillna(0.0)[['preschool_km', 'park_km', 'num_room']].values
trace1 = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=X[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = df['log_price'].values,

        colorscale = 'Portland',

        colorbar = dict(title = 'price_doc'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.5

    )

)



data=[trace1]

layout=dict(

    height=800,

    width=800,

    scene=go.Scene(

        xaxis=go.XAxis(title='preschool_km'),

        yaxis=go.YAxis(title='park_km'),

        zaxis=go.ZAxis(title='num_room')

    ),

    title='Prices by three dimensions'

)

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
X = df.fillna(0.0)[['industrial_km', 'full_sq', 'num_room']].values
trace1 = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=X[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = df['log_price'].values,

        colorscale = 'Portland',

        colorbar = dict(title = 'price_doc'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.5

    )

)



data=[trace1]

layout=dict(

    height=800,

    width=800,

    scene=go.Scene(

        xaxis=go.XAxis(title='industrial_km'),

        yaxis=go.YAxis(title='full_sq'),

        zaxis=go.ZAxis(title='num_room')

    ),

    title='Prices by three dimensions'

)

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
X = df.fillna(0.0)[['floor', 'full_sq', 'railroad_km']].values
trace1 = go.Scatter3d(

    x=X[:,0],

    y=X[:,1],

    z=X[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = df['log_price'].values,

        colorscale = 'Portland',

        colorbar = dict(title = 'price_doc'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.5

    )

)



data=[trace1]

layout=dict(

    height=800,

    width=800,

    scene=go.Scene(

        xaxis=go.XAxis(title='floor'),

        yaxis=go.YAxis(title='full_sq'),

        zaxis=go.ZAxis(title='railroad_km')

    ),

    title='Prices by three dimensions'

)

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')