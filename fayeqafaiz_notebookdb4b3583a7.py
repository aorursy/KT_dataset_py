

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot

import cufflinks as cf

init_notebook_mode(connected = True)

cf.go_offline()

%matplotlib inline



from plotly import tools

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.offline as offline

# Squarify for treemaps

import squarify

# Random for well, random stuff

import random

# operator for sorting dictionaries

import operator

# For ignoring warnings

import warnings

warnings.filterwarnings('ignore')







df = pd.read_csv("../input/ufc-data/data.csv")

df = pd.read_csv("../input/ufc-data/data.csv")

df.head()
df.info
df.tail()
df['B_Name']#names of the fighters in the blue corner
df['R_Name']#names of the fighters in the red corner
df.describe()
df.describe(include="all")
df.describe(include=['O'])

df.head(50)
print("Number of Blue fighters : ", len(df.B_ID.unique()))

print("Number of Red fighters : ", len(df.R_ID.unique()))
df.isnull().sum(axis=0)


df['B_Age'] = df['B_Age'].fillna(np.mean(df['B_Age']))

df['B_Height'] = df['B_Height'].fillna(np.mean(df['B_Height']))

df['R_Age'] = df['R_Age'].fillna(np.mean(df['R_Age']))

df['R_Height'] = df['R_Height'].fillna(np.mean(df['R_Height']))
from matplotlib import pyplot as plt

import numpy as np

fig = plt.figure(figsize=(18,7))

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

temp = df["winner"].value_counts()

ax.pie(temp.values, labels = temp.index,autopct='%1.2f%%',explode=(0,0.1,0,0),colors = ['red', 'k', 'lightskyblue', 'g'])

labels = temp.index

plt.legend(labels, loc="best")

plt.title('Most common way of winning a fight ')

plt.show()
#fig, ax = plt.subplots(1,2, figsize=(18, 9))

fig, ax = plt.subplots(1,2, figsize=(30,10))

sns.distplot(df.B_Age, ax=ax[0],color='cyan')

sns.distplot(df.R_Age, ax=ax[1],color='r')


BAge = df.groupby(['B_Age']).count()['winner']

BlueAge = BAge.sort_values(axis=0, ascending=False)

BlueAge.head(10)
RAge = df.groupby(['R_Age']).count()['winner']

RedAge = RAge.sort_values(axis=0, ascending=False)

RedAge.tail(10)
fig, ax = plt.subplots(1,2, figsize=(19, 7))

sns.distplot(df.B_Height, bins = 20, ax=ax[0]) #Blue 

sns.distplot(df.R_Height, bins = 20, ax=ax[1],color='red') #Red
fig, ax = plt.subplots(figsize=(14, 6))

sns.kdeplot(df.B_Height, shade=True, color='indianred', label='Red')

sns.kdeplot(df.R_Height, shade=True, label='Blue')

#NOT REQUIRED ?????????
df['Height Difference'] = df.B_Height - df.R_Height

df[['Height Difference', 'winner']].groupby('winner').mean()
fig = plt.figure(figsize=(18,7))

ax = fig.add_axes([0,0,1,1])

temp = df["winby"].value_counts()

labels = temp.index

sizes = temp.values

plt.pie(sizes, labels=labels, autopct='%.2f',colors = ['lightblue', 'yellow', 'yellowgreen',])



plt.legend(labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()


g = sns.FacetGrid(df, col='winby')

g.map(plt.hist, 'R_Age', bins=20,COLOR='R')
g = sns.FacetGrid(df, col='winby')

g.map(plt.hist, 'B_Age', bins=20,color='cyan')
cnt_srs = df['R_Location'].value_counts().head(15)



trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color=cnt_srs.values,

    ),

)



layout = go.Layout(

    title='Most popular training locations for Red fighters'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, filename="Ratio")
cnt_srs = df['B_Location'].value_counts().head(15)



trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color=cnt_srs.values,

    ),

)



layout = go.Layout(

    title='Most Popular training locations for Blue fighters'

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, filename="Ratio")
r1 = df[['B_Weight', 'B__Round1_Grappling_Reversals_Landed', 'B__Round1_Grappling_Standups_Landed', 'B__Round1_Grappling_Takedowns_Landed']].groupby('B_Weight').sum()



r1.plot(kind='line', figsize=(18,9), marker='o')

plt.grid(True)#if you do not mention this line the grids wont be there.

plt.show()

#A reversal is when a grappler maneuvers from underneath his or her opponent to gain a top position.

#A takedown is a technique that involves off-balancing an opponent and bringing him or her to the ground with the attacker landing on top

#Grappling standup is a position in which two standing individuals have grabbed ahold of one another where they can either throw strikes from this position or attemp for a takedown.
r5 = df[['B_Weight', 'B__Round5_Grappling_Reversals_Landed', 'B__Round5_Grappling_Standups_Landed', 'B__Round5_Grappling_Takedowns_Landed']].groupby('B_Weight').sum()



r5.plot(kind='line', figsize=(18,9), marker='o')

plt.grid(True)#if you do not mention this line the grids wont be there.

plt.show()
strk1 = df[['B_Weight', 'B__Round1_Strikes_Clinch Head Strikes_Landed', 'B__Round1_Strikes_Clinch Leg Strikes_Landed', 'B__Round1_Strikes_Clinch Body Strikes_Landed']].groupby('B_Weight').sum()





strk1.plot(kind='line', figsize=(18,9), marker='o')

plt.grid(True)#if you do not mention this line the grids wont be there.

plt.show()
strk5 = df[['B_Weight', 'B__Round5_Strikes_Clinch Head Strikes_Landed', 'B__Round5_Strikes_Clinch Leg Strikes_Landed', 'B__Round5_Strikes_Clinch Body Strikes_Landed']].groupby('B_Weight').sum()





strk5.plot(kind='line', figsize=(18,9), marker='o')

plt.grid(True)#if you do not mention this line the grids wont be there.

plt.show()
sns.lmplot(x="B__Round1_Strikes_Body Significant Strikes_Attempts", 

               y="B__Round1_Strikes_Body Significant Strikes_Landed", 

               col="winner", hue="winner", data=df, col_wrap=2, size=6)