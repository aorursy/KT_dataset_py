# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import plotly.graph_objects as go

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

df.drop(['show_id','description'], 'columns',inplace=True)
df['date_added'] = pd.to_datetime(df['date_added'])

df['month'] = df['date_added'].dt.month

df['Original year'] = df['release_year']

df['release_year'] = df['date_added'].dt.year



df['season'] = df.apply(lambda x: x['duration'].split()[0]  if "Season" in x['duration'] else "", axis =1)

df['duration'] = df.apply(lambda x: x['duration'].split()[0]  if not "Season" in x['duration'] else "", axis =1)
df.head()
group = df['type'].value_counts().reset_index()

labels = group['index']

sizes = group['type']

explode = (0.0,0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

plt.figure(figsize=(12, 3))

#Convert groupby into Dataframe

group = pd.DataFrame({'count' : df.groupby( [ "type", "release_year"] ).size()}).reset_index()

#Calcute percent

group['percent'] = group['count'].apply(lambda x: (x*100)/sum(group['count']))



#plot for movie

plt.plot(group[group['type']=='Movie']['release_year'],group[group['type']=='Movie']['count'],linestyle='-', marker='o', color='green', label="Movie")



#plot for TV Show

plt.plot(group[group['type']=='TV Show']['release_year'],group[group['type']=='TV Show']['count'], linestyle='-', marker='o',color='blue', label = "TV Show")

plt.title('Content added over the year')

plt.legend()
plt.figure(figsize=(12, 3))

#Convert groupby into Dataframe

group = pd.DataFrame({'count' : df.groupby( [ "type", "Original year"] ).size()}).reset_index()

#Calcute percent

group['percent'] = group['count'].apply(lambda x: (x*100)/sum(group['count']))



#plot for movie

plt.bar(group[group['type']=='Movie']['Original year'],group[group['type']=='Movie']['count'], label="Movie")



#plot for TV Show

plt.bar(group[group['type']=='TV Show']['Original year'],group[group['type']=='TV Show']['count'], label = "TV Show")

plt.title('Content added over the year')

plt.legend()
group  = pd.DataFrame({'count' : df.groupby( [ "type", "month"] ).size()}).reset_index()

plt.bar(group[group['type']=='Movie']['month'],group[group['type']=='Movie']['count'] , color = 'blue', edgecolor = 'black', label='Movie')

 

# Create cyan bars

plt.bar(group[group['type']=='TV Show']['month'],group[group['type']=='TV Show']['count'], color = 'cyan', edgecolor = 'black', label='TV Show')

plt.legend()

plt.title("In which month content added most")

plt.show()
df.sort_values(by=['Original year'])[df['type']=='Movie'].reset_index().loc[:15,['title', 'Original year']]
df.sort_values(by=['Original year'])[df['type']=='TV Show'].reset_index().loc[:10, ['title', 'Original year']]
df['Country'] = df['country'].dropna().str.split(',')

df['Country'] = df['Country'].apply(lambda x: next(iter(x)) if isinstance(x, list) else x)

data = pd.DataFrame({'Count': df.groupby('Country')['type'].count()}).reset_index()



colors = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",

    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",

    "#08519c", "#0b4083", "#08306b"

]





fig = go.Figure(data=go.Choropleth(

    locationmode = "country names",

    locations = data['Country'],

    z = data['Count'],

    colorscale = colors,

    autocolorscale=False,

    reversescale=False,

    colorbar_title = 'Countries with most content',

))



fig.update_layout(

    title_text='Countries with most content',

    geo=dict(

        showcoastlines=True,

    ),

)

fig.show()
import seaborn as sns

sns.distplot(df[df['type']=='Movie']['duration'], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
group  = pd.DataFrame({'Count' : df.groupby( [ "type", "season"] ).size()}).reset_index()

# Create cyan bars

plt.bar(group[group['type']=='TV Show']['season'],group[group['type']=='TV Show']['Count'], color = 'cyan', edgecolor = 'black', label='TV Show',)

plt.legend()

plt.title("TV Show with most seasons")

plt.show()
plt.figure(figsize=(9, 4))

group  = pd.DataFrame({'count' : df.groupby( [ "type", "rating"] ).size()}).reset_index()

plt.bar(group[group['type']=='Movie']['rating'],group[group['type']=='Movie']['count'] , color = 'blue', edgecolor = 'black', label='Movie')

 

# Create cyan bars

plt.bar(group[group['type']=='TV Show']['rating'],group[group['type']=='TV Show']['count'], color = 'cyan', edgecolor = 'black', label='TV Show')

plt.legend()

plt.title("Content Rating")

plt.show()
from collections import Counter

col = "listed_in"

categories = ", ".join(df['listed_in']).split(", ")

counter_list = Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Listed with", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
categories = ", ".join(df[df['country'] == 'India']['cast'].dropna()).split(',')

Cast = Counter(categories).most_common(20)

counterCast = [_[0] for _ in Cast ]

counterValue = [_[1] for _ in Cast ]

plt.bar( counterValue, counterCast, color = 'cyan', edgecolor = 'black', label='TV Show')

plt.title("India")
