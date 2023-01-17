import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.misc.pilutil import imread

%matplotlib inline
df = pd.read_csv('../input/deaths/kill_match_stats_final_0.csv')
df.head()
death_causes = df['killed_by'].value_counts()
sns.set_context('talk')

fig = plt.figure(figsize=(30, 10))
ax = sns.barplot(x=death_causes.index, y=[v / sum(death_causes) for v in death_causes.values])
ax.set_title('Rate of Death Causes')
ax.set_xticklabels(death_causes.index, rotation=90);
df_er = df[df['map']  == 'ERANGEL']
len(df_er)
df_er.head()
df_mr = df[df['map'] == 'MIRAMAR']
len(df_mr)
df_mr.head()
rank = 20

f, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].set_title('Death Causes Rate: Erangel (Top {})'.format(rank))
axes[1].set_title('Death Causes Rate: Miramar (Top {})'.format(rank))

counts_er = df_er['killed_by'].value_counts()
counts_mr = df_mr['killed_by'].value_counts()

sns.barplot(x=counts_er[:rank].index, y=[v / sum(counts_er) for v in counts_er.values][:rank], ax=axes[0] )
sns.barplot(x=counts_mr[:rank].index, y=[v / sum(counts_mr) for v in counts_mr.values][:rank], ax=axes[1] )
axes[0].set_ylim((0, 0.20))
axes[0].set_xticklabels(counts_er.index, rotation=90)
axes[1].set_ylim((0, 0.20))
axes[1].set_xticklabels(counts_mr.index, rotation=90);
f, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].set_title('Death Causes Rates: Erangel (1-{})'.format(rank//2))
axes[1].set_title('Death Causes Rate: Miramar (1-{})'.format(rank//2))

sns.barplot(x=counts_er[:rank//2].index, y=[v / sum(counts_er) for v in counts_er.values][:rank//2], ax=axes[0] )
sns.barplot(x=counts_mr[:rank//2].index, y=[v / sum(counts_mr) for v in counts_mr.values][:rank//2], ax=axes[1] )
axes[0].set_ylim((0, 0.20))
axes[0].set_xticklabels(counts_er[:rank//2].index, rotation=90)
axes[1].set_ylim((0, 0.20))
axes[1].set_xticklabels(counts_mr[:rank//2].index, rotation=90);
f, axes = plt.subplots(1, 2, figsize=(30, 10))
axes[0].set_title('Death Causes Rate: Erangel (9-{})'.format(rank))
axes[1].set_title('Death Causes Rate: Miramar (9-{})'.format(rank))

sns.barplot(x=counts_er[8:rank].index, y=[v / sum(counts_er) for v in counts_er.values][8:rank], ax=axes[0] )
sns.barplot(x=counts_mr[8:rank].index, y=[v / sum(counts_mr) for v in counts_mr.values][8:rank], ax=axes[1] )
axes[0].set_ylim((0, 0.05))
axes[0].set_xticklabels(counts_er[8:rank].index, rotation=90)
axes[1].set_ylim((0, 0.05))
axes[1].set_xticklabels(counts_mr[8:rank].index, rotation=90);
import math

def get_dist(df):
    dist = []

    for row in df.itertuples():
        subset = (row.killer_position_x - row.victim_position_x)**2 + (row.killer_position_y - row.victim_position_y)**2
        if subset > 0:
            dist.append(math.sqrt(subset) / 100)
        else:
            dist.append(0)
    return dist
df_dist = pd.DataFrame.from_dict({'dist(m)': get_dist(df_er)})
df_dist.index = df_er.index

df_er_dist = pd.concat([
    df_er,
    df_dist
], axis=1, sort=True)

df_er_dist.head()
df_er_dist[df_er_dist['dist(m)'] == max(df_er_dist['dist(m)'])]
df_dist = pd.DataFrame.from_dict({'dist(m)': get_dist(df_mr)})
df_dist.index = df_mr.index

df_mr_dist = pd.concat([
    df_mr,
    df_dist
], axis=1, sort=True)

df_mr_dist.head()
df_mr_dist[df_mr_dist['dist(m)'] == max(df_mr_dist['dist(m)'])]
df_er_dist['dist(m)'].describe()
df_mr_dist['dist(m)'].describe()
f, axes = plt.subplots(1, 2, figsize=(30, 10))

plot_dist = 150

axes[0].set_title('Engagement Dist. : Erangel')
axes[1].set_title('Engagement Dist.: Miramar')

plot_dist_er = df_er_dist[df_er_dist['dist(m)'] <= plot_dist]
plot_dist_mr = df_mr_dist[df_mr_dist['dist(m)'] <= plot_dist]

sns.distplot(plot_dist_er['dist(m)'], ax=axes[0])
sns.distplot(plot_dist_mr['dist(m)'], ax=axes[1]);
# Erangel image size = 4096
# Miramar image size = 1000
position = ['killer_position_x', 'killer_position_y', 'victim_position_x', 'victim_position_y']

for pos in position:
    df_er_dist[pos] = df_er_dist[pos]*4096/800000
    df_mr_dist[pos] = df_mr_dist[pos]*1000/800000
df_er_dist.head()
df_mr_dist.head()
match_id = df_er_dist['match_id'].unique()[0]
df_match = df_er_dist[df_er_dist['match_id'] ==  match_id]
df_match['time'].describe()
def plot_engagement_pos(df_match):
    f, axes = plt.subplots(1, 3, figsize=(30, 30))
    bg = imread('../input/erangel.jpg')

    time_25 = df_match['time'].describe()['25%']
    time_50 = df_match['time'].describe()['50%']
    time_75 = df_match['time'].describe()['75%']
    time_max = df_match['time'].describe()['max']

    axes[0].imshow(bg)
    axes[0].set_title('Engagement Position: Erangle (0s - 720s)')
    df_match_1 = df_match[df_match['time'] <= time_25]
    x = df_match_1['killer_position_x'].values
    y = df_match_1['killer_position_y'].values
    u = df_match_1['victim_position_x'].values
    v = df_match_1['victim_position_y'].values
    axes[0].quiver(x, y, u, v, angles='xy', scale_units='xy', scale=25, color='red')

    axes[1].imshow(bg)
    axes[1].set_title('Engagement Position: Erangle (720s - 1421s)')
    df_match_2 = df_match[(df_match['time'] >=  time_25) & (df_match['time'] <= time_50)]
    x = df_match_2['killer_position_x'].values
    y = df_match_2['killer_position_y'].values
    u = df_match_2['victim_position_x'].values
    v = df_match_2['victim_position_y'].values
    axes[1].quiver(x, y, u, v, angles='xy', scale_units='xy', scale=25, color='red');

    axes[2].imshow(bg)
    axes[2].set_title('Engagement Position: Erangle (1421s - 1885s)')
    df_match_2 = df_match[(df_match['time'] >= time_50) & (df_match['time'] <= time_max)]
    x = df_match_2['killer_position_x'].values
    y = df_match_2['killer_position_y'].values
    u = df_match_2['victim_position_x'].values
    v = df_match_2['victim_position_y'].values
    axes[2].quiver(x, y, u, v, angles='xy', scale_units='xy', scale=25, color='red');
x =df_match['killer_position_x'].values
y = df_match['killer_position_y'].values

u = df_match['victim_position_x'].values
v = df_match['victim_position_y'].values

bg = imread('../input/erangel.jpg')
fig, ax = plt.subplots(1, 1, figsize=(15,15))
ax.imshow(bg)
ax.set_title('Engagement Position Map: Erangle')
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=25, color='red');
plot_engagement_pos(df_match)
match_id = df_er_dist['match_id'].unique()[1]
df_match = df_er_dist[df_er_dist['match_id'] ==  match_id]
df_match['time'].describe()
x =df_match['killer_position_x'].values
y = df_match['killer_position_y'].values

u = df_match['victim_position_x'].values
v = df_match['victim_position_y'].values

bg = imread('../input/erangel.jpg')
fig, ax = plt.subplots(1, 1, figsize=(15,15))
ax.imshow(bg)
ax.set_title('Engagement Position Map: Erangle')
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=25, color='red');
plot_engagement_pos(df_match)
