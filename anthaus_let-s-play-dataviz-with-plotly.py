# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hostel-world-dataset/Hostel.csv')

df
print(len(df))

print(df.isna().sum())

df.describe()
for idx, row in df.iterrows():

    c = 0

    if "km from city centre" not in row['Distance']:

        c += 1

print(c)
dist = []

for d in df['Distance']:

    dist.append(d.replace('km from city centre',''))

dist[:5]
del df['Distance']

df['Distance'] = dist

df.head(5)
set(df['rating.band'])
fabulous = df[df['rating.band'] == 'Fabulous']

good = df[df['rating.band'] == 'Good']

rating = df[df['rating.band'] == 'Rating']

superb = df[df['rating.band'] == 'Superb']

verygood = df[df['rating.band'] == 'Very Good']
fig = px.box(df, x="rating.band", y="summary.score")

fig.show()
roten = []

for idx, row in df.iterrows():

    if row['rating.band'] is np.nan:

        roten.append(idx)

df_corr = df.drop(roten)
import plotly.graph_objects as go

X = ('Rating', 'Good', 'Very Good', 'Fabulous', 'Superb')

#X = set(df_corr["rating.band"])

Y = [len(df_corr[df_corr["rating.band"] == x]) for x in X]

fig = go.Figure([

    go.Bar(x=X, y=Y)

])



fig.show()
fig = px.scatter(df_corr, x="Distance", y="summary.score", color="rating.band")

fig.show()
fig = px.scatter_3d(df_corr, x="atmosphere", y="facilities", z="staff", color="rating.band")

fig.show()
roten = []

for idx, row in df_corr.iterrows():

    if row['lat'] is np.nan or row['lon'] is np.nan:

        roten.append(idx)

df_corr2 = df_corr.drop(roten)
fig = px.scatter_geo(df_corr2, lat='lat', lon='lon', hover_name='hostel.name', color='rating.band', projection="natural earth", center={'lat':35.42, 'lon':139.43}, scope='asia')

fig.show()
fig = px.scatter_mapbox(df_corr2, lat='lat', lon='lon', hover_name='hostel.name', color='rating.band', zoom=15)

fig.show()