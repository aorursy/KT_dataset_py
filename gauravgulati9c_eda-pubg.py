import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

%matplotlib inline
import os
import warnings
warnings.filterwarnings("ignore")
print(os.listdir())
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
sample = pd.read_csv('../input/sample_submission_V2.csv')
train.head()
train.info()
train.isnull().any()
sns.set_style('dark')
print("On an average a player uses {:.2f} number of healing items in his/her gameplay.".format(np.mean(train.heals.values)))
print("90% Players use {:.2f} number of healing items in his/her gameplay.".format((train.heals.quantile(0.9))))
train.head()
print("% Distribution of kills of many players")
(train.kills.value_counts() / sum(train.kills) * 100)[:10]
temp = train.copy()
def kill_dist(x):
    if x < 15:
        return x
    else:
        return "15+"
temp["kills"] = temp["kills"].apply(kill_dist)
temp["kills"].unique()
print(temp.shape)
print(train.shape)
print(test.shape)
print(sample.shape)
temp.columns
trace1 = go.Bar(
            x=temp['kills'].value_counts().index,
            y=temp['kills'].value_counts().values,
            marker = dict(color = 'rgba(255, 255, 135, 1)',
                  line=dict(color='rgb(0,0,255)',width=2)),

            name = 'Kills'
    )

trace2 = go.Bar(
            x=train.heals.value_counts()[:10].index,
            y=train.heals.value_counts()[:10].values,
            marker = dict(color = 'rgba(255, 128, 128, 3)',
                      line=dict(color='rgb(0,0,255)',width=2)),
            name='Heals'
    )

data = [trace1, trace2]

layout = dict(title = 'Kills Count Plot',
              xaxis= dict(title= 'Kills v/s Heals',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Number")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
temp2 = train.copy()
temp2['CategoryKills'] = pd.cut(train['kills'], [-1, 0, 2, 5, 10, 50, 100],
      labels=['0 kills','1-2 kills', '2-4 kills', '5-10 kills', '10-50', '> 50 kills'])
train.head()
temp2['CategoryDamageDealt'] = pd.cut(train['damageDealt'], [-1, 0, 10, 50, 150, 300, 1000, 6000],
      labels = ['O Damage Taken', '1-10 Damage Taken', '11-50 Damage Taken', '51-150 Damage Taken', '151-300 Damage Taken', '301-1000 Damage Taken', '1000+ Damage Taken']) 
plt.figure(figsize=(16, 8))
sns.countplot(temp2['CategoryDamageDealt'], saturation = 0.76,
              linewidth=2,
              edgecolor = sns.set_palette("dark", 3))
plt.xlabel("Damage Taken")
plt.ylabel("Number")
plt.figure(figsize=(16, 8))
sns.boxplot(x='CategoryDamageDealt', y='winPlacePerc', data=temp2, palette='Set2', saturation=0.8, dodge=True, linewidth=2.5)
plt.xlabel("Damage Dealt")
plt.ylabel("Win Place Percentage")
plt.title('Damage and Win Place Percentage Distribution')
plt.figure(figsize=(16, 8))
sns.boxplot(x='CategoryKills', y='winPlacePerc', data=temp2, palette='Set3', saturation=0.8, linewidth=2.5)
plt.xlabel("Kills Distribution")
plt.ylabel("Win Place Percentage")
plt.title("Category Kills and Win Percentage Dependencies")
temp2['CategoryweaponsAcquired'] = pd.cut(train['weaponsAcquired'], [-1, 0, 5, 10, 15, 20, 100],
      labels = ['O weapons', '1-5 weapons', '6-10 weapons', '11-15 weapons', '16-20 weapons', '20+ weapons']) 
train.head()
trace1 = go.Bar(
            x=temp2['CategoryweaponsAcquired'].value_counts().index,
            y=temp2['CategoryweaponsAcquired'].value_counts().values,
            marker = dict(
                  line=dict(color='rgb(0,0,255)',width=2)),
            name = 'Weapons Acquired'
    )

data = [trace1]

layout = dict(title = 'Weapons Acquired Plot',
              xaxis= dict(title= 'Weapons Acquired',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Number")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
train.head()
plt.figure(figsize=(10, 8))
train.matchType.value_counts().plot(kind='barh', align='center')
plt.title("Match Types")
plt.xlabel("Count")
distances = train[['rideDistance', 'swimDistance', 'walkDistance']]
distances['Total Distance'] = distances['rideDistance'] + distances['swimDistance'] + distances['walkDistance']
plt.figure(figsize=(10, 6))
# plt.hist(distances['Total Distance'], bins=20)
sns.distplot(distances['Total Distance'], bins=10)
plt.title("Total Distance Distribution")
plt.xlabel("Distribution")
trace1 = go.Bar(
            x=temp2['vehicleDestroys'].value_counts().index,
            y=temp2['vehicleDestroys'].value_counts().values,
            marker = dict(
                color='rgb(102,149,232)',
                  line=dict(color='rgb(0,0,100)',width=2)),
            name = 'Vehicles Destroyed'
    )

data = [trace1]

layout = dict(title = 'Vehicles Destroyed',
              xaxis= dict(title= 'Vehicles',ticklen= 5,zeroline= False),
              yaxis = dict(title = "Number")
             )
fig = dict(data = data, layout=layout)
iplot(fig)
plt.figure(figsize=(12, 8))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=temp2)
plt.xlabel('Number of Vehicle Destroys')
plt.ylabel('Win Percentage')
plt.title('Vehicle Destroys affecting Win Ratio')
plt.show()
### Vehicles destroyed along with Weapons acquired affecting Win Percentage
plt.figure(figsize=(12, 8))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=temp2, hue="CategoryweaponsAcquired")
plt.xlabel('Number of Vehicle Destroys')
plt.ylabel('Win Percentage')
plt.title('Vehicle Destroys affecting Win Ratio')
plt.show()