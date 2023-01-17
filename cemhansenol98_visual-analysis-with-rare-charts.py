import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from collections import Counter
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

from wordcloud import WordCloud

import squarify
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv("../input/us-police-shootings/shootings.csv")

data.head()
data['date'] = pd.to_datetime(data['date'])

data['year'] = data['date'].dt.year

data['month'] = data['date'].dt.month

data['day'] = data['date'].dt.day

data['day_of_week'] = data['date'].dt.dayofweek
data.shape
data['above30'] = ['above30'if i >=30 else 'below30'for i in data.age]



data['generation']="-"

for i in data.age.index:

    if data['age'][i] >=0 and data['age'][i]<30:

        data['generation'][i]='20s'

    elif data['age'][i] >=30 and data['age'][i]<40:

        data['generation'][i]='30s'

    elif data['age'][i] >=40 and data['age'][i]<50:

        data['generation'][i]='40s'

    elif data['age'][i] >=50 and data['age'][i]<60:

        data['generation'][i]='50s'

    else:

        data['generation'][i]='60+'
state_names = dict(data['state'].value_counts()).keys()

state_cases = data['state'].value_counts().tolist()



state_df = pd.DataFrame()

state_df['state'] = state_names

state_df['incidents'] = state_cases

state_df.head()
state_count = Counter(data.state)

most_common_states = state_count.most_common(15)

x,y = zip(*most_common_states)

x,y= list(x), list(y)



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y, palette=sns.cubehelix_palette(len(x)))

plt.title('Most Common 15 States ')
generation_count = Counter(data.generation)

generations = generation_count.most_common(5)

x,y = zip(*generations)

x,y= list(x), list(y)



plt.figure(figsize=(15,10))

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]

ax=sns.barplot(x=x,y=y, palette=sns.color_palette(flatui))

plt.title('Most Common 15 States ')
df = px.data.tips()

fig = px.sunburst(data, path=['year', 'race', 'gender'], values='age',title="Dont Forget to Click Chart to Examine Deeply ")

fig.show()
targets = data['arms_category'].value_counts().tolist()

values = list(dict(data['arms_category'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Arms Categories',

    color_discrete_sequence=['darkcyan', 'lawngreen']

)

fig.show()
g = sns.catplot(x="gender", y="age",

                hue="manner_of_death", col="year",

                data=data, kind="box",

                height=7, aspect=.7);

sns.swarmplot(x='gender',y='age',hue='manner_of_death',data=data)

plt.show()
sns.countplot(data.gender)

sns.countplot(data.manner_of_death)

plt.show()
sns.countplot(x=data.above30,  palette="Set2")

plt.ylabel('Number of People')

plt.title('Age of People', color='blue', fontsize=15)
sns.violinplot(x ='year', y ='age', data = data, hue ='gender', split = True)
sns.stripplot(x ='race', y ='age', data = data,

              jitter = True, hue ='flee', dodge = True)
plt.figure(figsize = (20, 12))

squarify.plot(sizes = data.state.value_counts().values, alpha = 0.8,

              label = data.state.unique())

plt.title('State - People', fontsize = 20)

plt.axis('off')

plt.show()
fig = go.Figure(data=go.Choropleth(

    locations=state_df['state'], 

    z = state_df['incidents'].astype(int), 

    locationmode = 'USA-states',

    colorscale = 'Viridis', 

    colorbar_title = "Incidents Density", 

))



fig.update_layout(

    title_text = 'Police Shooting Cases Across US States', 

    geo_scope='usa', 

)



fig.show()
x2020=data.city[data.year==2020]

plt.subplots(figsize=(10,19))

wordcloud = WordCloud(background_color = 'white',

                     width=512,

                     height=384).generate("".join(x2020))



plt.imshow(wordcloud)

plt.axis('off')