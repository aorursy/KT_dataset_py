# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly.express as px

import plotly.figure_factory as ff

from wordcloud import WordCloud, ImageColorGenerator

import plotly.graph_objects as go



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv')



state_population= pd.read_csv("/kaggle/input/us-population-by-state-2019/US_State_Population.csv", usecols=['State Code','July 2019 Estimate'])
data.info()
data.sample(10)
data['manner_of_death'].unique()
print(f"Unique values provided in ARMED field:\n\n {data['armed'].unique()}\n\n")

print(f"Unique values provided in ARMS CATEGORY field:\n\n {data['arms_category'].unique()}\n\n")

print(f"Unique values provided in THREAT LEVEL field:\n\n {data['threat_level'].unique()}\n\n")

print(f"Unique values provided in MANNER OF DEATH field:\n\n {data['manner_of_death'].unique()}\n\n")
data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) # convert the string to datetime

data['count'] = 1 # this helps with grouping incidents and provides total instance count



data['armed'] = data['armed'].str.replace(" ","_") # I want to keep the words that are in the same field together in the word cloud e.g: box cutter -> box_cutter

data['armed'] = data['armed'].str.replace("_and_"," ") # Unless those words represent multiple arms e.g.: gun_and_knife
armed_text = " ".join(text for text in data['armed'])

wordcloud = WordCloud(background_color="white").generate(armed_text)

plt.figure(figsize=[10,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title("VICTIMS WERE ARMED WITH:")

plt.show()
fig = px.histogram(data, x="date", y="count", color="gender", marginal="rug", template='ggplot2', title='US Police Shootings Timeline')

fig.update_layout(yaxis=dict(title_text="Count Of Shooting"),xaxis=dict(title_text='Date of Occurance'))

fig.show()
fig = px.pie(data, values='count', names='race', title='Demographics of Victims')



fig.show()
fig = px.histogram(data, x="state", y="count", color="gender", title="Shootings By State").update_xaxes(categoryorder="total descending")

fig.update_layout(yaxis=dict(title_text="Count Of Shooting"),xaxis=dict(title_text='State of Incident'))

fig.update_layout(barmode='group')

fig.show()
shooting_per_state = pd.DataFrame(data.groupby(by='state').sum()['count']).rename(columns={"count":"Total Shootings"})



shooting_per_state = shooting_per_state.merge(state_population, left_on='state', right_on='State Code').rename(columns={"July 2019 Estimate":"Population"})



shooting_per_state['Shooting Per Capita'] = shooting_per_state['Total Shootings']/shooting_per_state['Population']*100000
fig = go.Figure(data=go.Choropleth(

    locations=shooting_per_state['State Code'], # Spatial coordinates

    z = shooting_per_state['Shooting Per Capita'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Reds',

))



fig.update_layout(

    title_text = 'US Police Shootings by State per Capita (100.000)',

    geo_scope='usa', # limit map scope to USA

)



fig.show()
f, ax = plt.subplots(figsize=(15, 10))

sns.set_style("dark")

sns.countplot(x="threat_level", hue="body_camera",palette="pastel", edgecolor="0.5", data=data);

ax.set(xlabel='Threat Level',ylabel='Shooting Count')
f, ax = plt.subplots(figsize=(15, 10))

sns.set_style("dark")

sns.countplot(x="signs_of_mental_illness", hue="race",palette="pastel", edgecolor="0.5", data=data);

ax.set(xlabel='Signs of Mentall Illness',ylabel='Shooting Count')
f, ax = plt.subplots( figsize=(20, 10))

sns.swarmplot(x="race", y="age", data=data, ax=ax)

ax.set(xlabel='Race of Victim',ylabel='Age of Victim')