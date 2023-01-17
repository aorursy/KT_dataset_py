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
train['damageDealt'].min()
temp2['CategoryDamageDealt'] = pd.cut(train['damageDealt'], [-1, 0, 10, 50, 150, 300, 1000, 6000],
      labels = ['O Damage Taken', '1-10 Damage Taken', '11-50 Damage Taken', '51-150 Damage Taken', '151-300 Damage Taken', '301-1000 Damage Taken', '1000+ Damage Taken']) 
plt.figure(figsize=(16, 8))
sns.countplot(temp2['CategoryDamageDealt'], saturation = 0.76,
              linewidth=2,
              edgecolor = sns.set_palette("dark", 3))
plt.xlabel("Damage Taken")
plt.ylabel("Number")
plt.figure(figsize=(16, 8))
sns.boxplot(x='CategoryKills', y='winPlacePerc', data=temp2, palette='Set3', saturation=0.8, linewidth=2.5)
plt.xlabel("Kills Distribution")
plt.ylabel("Win Place Percentage")
plt.title("Category Kills and Win Percentage Dependencies")
plt.figure(figsize=(16, 8))
sns.boxplot(x='CategoryDamageDealt', y='winPlacePerc', data=temp2, palette='Set2', saturation=0.8, dodge=True, linewidth=2.5)
plt.xlabel("Damage Dealt")
plt.ylabel("Win Place Percentage")
plt.title('Damage and Win Place Percentage Distribution')