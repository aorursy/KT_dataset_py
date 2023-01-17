# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/earthquake/earthquake.csv')
df.fillna(-1,inplace=True)

df
df_corr =df.corr()

sns.heatmap(df_corr)

df.country.value_counts()

x = df.country.value_counts().index

y = df.country.value_counts().values 

f, (ax1) = plt.subplots(figsize=(15, 30), sharex=True)

sns.barplot(x=x, y=y, palette="rocket", ax=ax1)

plt.xticks(rotation = 90 , color='white')

ax1.axhline(0, color="k", clip_on=False)



bool_turkey = df.country == 'turkey'

turkey = df[bool_turkey]

turkey.city.value_counts()

f, (ax1) = plt.subplots(figsize=(16, 16), sharex=True)

sns.barplot(x=turkey.city.value_counts().index, y=turkey.city.value_counts().values, palette="ch:2.5,-.2,dark=.3", ax=ax1)

plt.xticks(rotation = 90 , color='black')

ax1.axhline(0, color="k", clip_on=False)







import plotly.express as px

turkey.direction

ax = sns.barplot(x=turkey.direction.value_counts().index, y=turkey.direction.value_counts().values,palette="ch:2.5,-.2,dark=.3" )

plt.xticks(rotation=90 , color ='black')

plt.yticks(rotation=0 , color ='black')

plt.show()



















fig = px.scatter_mapbox(turkey, lat="lat", lon="long", hover_name="city", hover_data=["depth", "richter",'direction'],

                        color_discrete_sequence=["light green"], zoom=5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
turkey['date'] = pd.to_datetime(turkey['date'])

turkey['year'] = turkey['date'].dt.year

labels_turkey = turkey.year.value_counts().index

plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(turkey.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(turkey.year.value_counts().index) ,y = turkey.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()

boolen_mediterranean = df['country'] == 'mediterranean'

mediterranean = df[boolen_mediterranean]

mediterranean
mediterranean['date'] = pd.to_datetime(mediterranean['date'])

mediterranean['year'] = mediterranean['date'].dt.year

plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(mediterranean.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(mediterranean.year.value_counts().index) ,y = mediterranean.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()



fig = px.scatter_mapbox(mediterranean, lat="lat", lon="long", hover_name="city", hover_data=["depth", "richter",'direction'],

                        color_discrete_sequence=["light green"], zoom=5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
boolen_greece = df['country'] == 'greece'

greece = df[boolen_greece]

greece
plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(greece.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(greece.year.value_counts().index) ,y = greece.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()

greece_value_counts =greece.area.value_counts().index[1:]

greece_value_counts
f, (ax1) = plt.subplots(figsize=(16, 16), sharex=True)

sns.barplot(x=greece.area.value_counts().index[1:], y=greece.area.value_counts().values[1:], palette="ch:2.5,-.2,dark=.3", ax=ax1)

plt.xticks(rotation = 90 , color='black')

ax1.axhline(0, color="k", clip_on=False)

fig = px.scatter_mapbox(greece, lat="lat", lon="long", hover_name="city", hover_data=["depth", "richter",'direction'],

                        color_discrete_sequence=["light green"], zoom=5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year

labels = df.year.value_counts().index

plt.figure(figsize=(25,25))

pal = sns.cubehelix_palette(len(df.year.value_counts().index), start=.5, rot=-.75)

sns.barplot(x=np.sort(df.year.value_counts().index) ,y = df.year.value_counts().values,palette =pal )

plt.xticks(rotation=90 , color ='black')

plt.show()

long_data = zip(df.dist,df.depth,df.xm,df.md,df.richter,df.mw,df.ms,df.mb)

data_long = pd.DataFrame(long_data,columns=["dist","depth","xm","md","richter","mw","ms","mb"])

data_long.head()

data_long.dist = data_long.dist /max(data_long.dist)

data_long.depth = data_long.depth /max(data_long.depth)

data_long.xm = data_long.xm /max(data_long.xm)

data_long.md = data_long.md /max(data_long.md)

data_long.richter = data_long.richter /max(data_long.richter)

data_long.mw = data_long.mw /max(data_long.mw)

data_long.ms = data_long.ms /max(data_long.ms)

data_long.mb = data_long.mb /max(data_long.mb)



plt.figure(figsize = (32,32))

pal = sns.cubehelix_palette(8, start=.5, rot=-.75)

sns.violinplot(data = data_long,palette = pal , inner = "points" )
