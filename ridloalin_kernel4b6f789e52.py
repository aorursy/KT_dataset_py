import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
injury = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

playlist = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

playtrack = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
playtrack.head(5)
injury.head(5)
playlist.head(5)
injury_playkey = pd.DataFrame(injury.PlayKey.value_counts()).index

playlist_injury = pd.DataFrame()

for i in injury_playkey:

    new_table = playlist[playlist.PlayKey == i]

    playlist_injury = pd.concat([playlist_injury, new_table])
col = ['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42']

test = pd.DataFrame()

for c in col: test[c] = injury[c]
bodypart = pd.DataFrame(injury.BodyPart.value_counts())

surface = pd.DataFrame(injury.Surface.value_counts())

period = []



for i in range(len(injury)):

    x = test.loc[i].sum()

    for j in range(4):

        if x == (j+1): period.append(col[j])



period = pd.DataFrame(pd.DataFrame(period, columns=['period']).period.value_counts())

period = period.reindex(col)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))



sns.barplot(x=bodypart.index, y=bodypart.BodyPart, ax=ax1)

sns.barplot(x=surface.index, y=surface.Surface, ax=ax2)

sns.barplot(x=period.index, y=period.period, ax=ax3)
roster = pd.DataFrame(playlist.RosterPosition.value_counts())

position = pd.DataFrame(playlist.Position.value_counts())

positiongroup = pd.DataFrame(playlist.PositionGroup.value_counts())



injury_roster = pd.DataFrame(playlist_injury.RosterPosition.value_counts())

injury_position = pd.DataFrame(playlist_injury.Position.value_counts())

injury_group = pd.DataFrame(playlist_injury.PositionGroup.value_counts())



import seaborn as sns

import matplotlib.pyplot as plt



f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))



sns.barplot(x=injury_roster.index, y=injury_roster.RosterPosition, data=injury_roster, ax=ax1)

sns.barplot(x=injury_position.index, y=injury_position.Position, data=injury_position, ax=ax2)

sns.barplot(x=injury_group.index, y=injury_group.PositionGroup, data=injury_group, ax=ax3)
offdef = {}



offdef['offense'] = ['Wide Receiver', 'Running Back', 'Offensive Lineman', 'Tight End']

offdef['defense'] = ['Linebacker', 'Safety', 'Defensive Lineman', 'Cornerback']



off_def = [] 



for o in ['offense', 'defense']:

    no = []

    for j in range(4):

        no.append(injury_roster.RosterPosition.loc[offdef[o][j]])

    no = np.array(no).sum()

    off_def.append(no)



sns.barplot(x=['Offense', 'Defense'], y=off_def)
injury_playtrack = pd.DataFrame()

for i in injury_playkey:

    injury_playtrack = pd.concat([injury_playtrack, playtrack[playtrack.PlayKey == i]])



no = np.linspace(0, 76, 8)

for i, n in enumerate(no): no[i] = int(round(n))



speed = {}



for i in range(7):

    speed[str(i)] = pd.DataFrame()

    for j in np.arange(no[i], no[i+1]):

        j = int(j)

        table = injury_playtrack[injury_playtrack.PlayKey == injury_playkey[j]].tail(40)

        table['time'] = np.arange(0, 4, 0.1)

        speed[str(i)] = pd.concat([speed[str(i)], table])    
time = np.arange(0, 40)

gap = [(no[i+1] - no[i]) for i in range(7)]

f, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(7, 1, figsize=(15, 60))

axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]



for i in range(7):

    df = speed[str(i)]

    sns.lineplot(x='time', y='s', hue='PlayKey', palette='deep', data=df, ax=axes[i])
width = np.arange(0, 130, 10)

num = ['G', '10', '20', '30', '40', '50', '40', '30', '20', '10', 'G']



new_width = list(width)

new_width.remove(new_width[0])

new_width.remove(new_width[11])



def plot_position(k):

    plt.figure(figsize=(13, 5.5))

    used = injury_playtrack[injury_playtrack.PlayKey == k]



    for w in width:

        plt.plot([w, w], [0, 53.3], color='black')

    plt.plot([0, 120], [53.3, 53.3], color='black')

    plt.plot([0, 120], [0, 0], color='black')

    for n, w in zip(num, new_width):

        plt.text(w-2, 3, n, fontsize=20)

        plt.text(w-2, 50, n, fontsize=20, rotation=180)

    plt.text(2, 40, 'HOME ENDZONE', rotation=90, fontsize=20)

    plt.text(115, 40, 'AWAY ENDZONE', rotation=270, fontsize=20)

    sns.scatterplot(x='x', y='y', data=used, hue='time', s=200)

    plt.show()
plot_position(injury_playkey[20])
plot_position(injury_playkey[31])
injury_true = []

for x in list(playlist.PlayKey): injury_true.append(int(x in injury_playkey))



def get_heatmap(param):

    oh = OneHotEncoder()

    enc = LabelEncoder()

    first = enc.fit_transform(list(playlist[param]))

    secnd = oh.fit_transform(first.reshape([-1, 1]))

    df = pd.DataFrame(secnd.toarray())

    df.columns = enc.classes_

    df['injury'] = injury_true

    corr = df.corr()

    plt.figure(figsize=(20, 20))

    sns.heatmap(corr, linewidths=0.3)

    corr = corr.drop('injury')

    plt.figure(figsize=(20, 20))

    chart = sns.barplot(x=corr.index, y=corr.injury)    

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
get_heatmap('RosterPosition')
get_heatmap('PlayerDay')
get_heatmap('PlayerGame')
get_heatmap('StadiumType')
get_heatmap('FieldType')
get_heatmap('Weather')
get_heatmap('PlayType')
get_heatmap('PlayerGamePlay')