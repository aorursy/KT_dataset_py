import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# from zipfile import ZipFile

import os

print(os.listdir("../input"))
print(os.listdir("../input/aggregate"))

print(os.listdir('../input/deaths'))
df0 = pd.read_csv("../input/aggregate/agg_match_stats_0.csv")

# df1 = pd.read_csv("../input/aggregate/agg_match_stats_1.csv")

# df2 = pd.read_csv("../input/aggregate/agg_match_stats_2.csv")

# df3 = pd.read_csv("../input/aggregate/agg_match_stats_3.csv")

# df4 = pd.read_csv("../input/aggregate/agg_match_stats_4.csv")

# dflist = [df0,df1,df2,df3,df4]

# df = pd.concat(dflist)

df = df0

df = df.dropna()

df.drop_duplicates(inplace=True)

df.info()
# 添加是否成功吃鸡列

df['won'] = df['team_placement'] == 1
# 添加是否搭乘过车辆列

df['drove'] = df['player_dist_ride'] != 0
df.loc[df['player_kills'] < 40, ['player_kills', 'won']].groupby('player_kills').won.mean().plot.bar(figsize=(15,6), rot=0)

plt.xlabel('Kill_#', fontsize=12)

plt.ylabel("Win_Ratio", fontsize=12)

plt.title('correlation of Kill and Win', fontsize=12)
g = sns.FacetGrid(df.loc[df['player_kills']<=10, ['party_size', 'player_kills']], row="party_size", size=4, aspect=2)

g = g.map(sns.countplot, "player_kills")
df.loc[df['party_size']!=1, ['player_assists', 'won']].groupby('player_assists').won.mean().plot.bar(figsize=(15,6), rot=0)

plt.xlabel('Assist_#', fontsize=12)

plt.ylabel("Win_Ratio", fontsize=12)

plt.title('Correlation of Assist and Win', fontsize=12)
df.groupby('drove').won.mean().plot.barh(figsize=(6,3))

plt.xlabel("Win_Ratio", fontsize=12)

plt.ylabel("Pass By Car", fontsize=12)

plt.title('Correlation of Pass By Car and Win_ratio', fontsize=12)

plt.yticks([1,0],['Y','N'])
dist_ride = df.loc[df['player_dist_ride']<12000, ['player_dist_ride', 'won']]
labels=["0-1k", "1-2k", "2-3k", "3-4k","4-5k", "5-6k", "6-7k", "7-8k", "8-9k", "9-10k", "10-11k", "11-12k"]

dist_ride['drove_cut'] = pd.cut(dist_ride['player_dist_ride'], 12, labels=labels)
dist_ride.groupby('drove_cut').won.mean().plot.bar(rot=60, figsize=(8,4))

plt.xlabel("Drive_Distance", fontsize=12)

plt.ylabel("Win_Ratio", fontsize=12)

plt.title('Correlation of Drive_dis and Win_ratio', fontsize=12)
match_unique = df.loc[df['party_size'] == 1, 'match_id'].unique()
dfd0 = pd.read_csv("../input/deaths/kill_match_stats_final_0.csv")

# dfd1 = pd.read_csv("../input/deaths/kill_match_stats_final_1.csv")

# dfd2 = pd.read_csv("../input/deaths/kill_match_stats_final_2.csv")

# dfd3 = pd.read_csv("../input/deaths/kill_match_stats_final_3.csv")

# dfd4 = pd.read_csv("../input/deaths/kill_match_stats_final_4.csv")

# dfdlist = [dfd0,dfd1,dfd2,dfd3,dfd4]

# dfd = pd.concat(dfdlist)

dfd = dfd0

dfd = dfd.dropna()

dfd.drop_duplicates(inplace=True)

dfd.info()
dfd_solo = dfd[dfd['match_id'].isin(match_unique)]
death_180_seconds_erg = dfd_solo.loc[(dfd_solo['map'] == 'ERANGEL')&(dfd_solo['time'] < 180)&(dfd_solo['victim_position_x']>0), :].dropna()

death_180_seconds_mrm = dfd_solo.loc[(dfd_solo['map'] == 'MIRAMAR')&(dfd_solo['time'] < 180)&(dfd_solo['victim_position_x']>0), :].dropna()
# 选择存活不过180秒的玩家死亡位置

data_erg = death_180_seconds_erg[['victim_position_x', 'victim_position_y']].values

data_mrm = death_180_seconds_mrm[['victim_position_x', 'victim_position_y']].values
# 重新scale玩家位置

data_erg = data_erg*4096/800000

data_mrm = data_mrm*1000/800000
from scipy.ndimage.filters import gaussian_filter

import matplotlib.cm as cm

from matplotlib.colors import Normalize

# from scipy.misc.pilutil import imread

from imageio import imread
def heatmap(x, y, s, bins=100):

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

    heatmap = gaussian_filter(heatmap, sigma=s)



    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return heatmap.T, extent



bg = imread('../input/erangel.jpg')

hmap, extent = heatmap(data_erg[:,0], data_erg[:,1], 4.5)

alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*4.5, 0.0, 1.)

colors = Normalize(0, hmap.max(), clip=True)(hmap)

colors = cm.Reds(colors)

colors[..., -1] = alphas



fig, ax = plt.subplots(figsize=(24,24))

ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)

ax.imshow(bg)

ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)

plt.gca().invert_yaxis()
bg = imread('../input/miramar.jpg')

hmap, extent = heatmap(data_mrm[:,0], data_mrm[:,1], 4)

alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*4, 0.0, 1.)

colors = Normalize(0, hmap.max(), clip=True)(hmap)

colors = cm.Reds(colors)

colors[..., -1] = alphas



fig, ax = plt.subplots(figsize=(24,24))

ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)

ax.imshow(bg)

ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)

#plt.scatter(plot_data_mr[:,0], plot_data_mr[:,1])

plt.gca().invert_yaxis()
#毒圈

death_final_circle_erg = dfd_solo.loc[(dfd_solo['map'] == 'ERANGEL')&(dfd_solo['victim_placement'] == 2)&(dfd_solo['victim_position_x']>0)&(dfd_solo['killer_position_x']>0), :].dropna()

death_final_circle_mrm = dfd_solo.loc[(dfd_solo['map'] == 'MIRAMAR')&(dfd_solo['victim_placement'] == 2)&(dfd_solo['victim_position_x']>0)&(dfd_solo['killer_position_x']>0), :].dropna()
final_circle_erg = np.vstack([death_final_circle_erg[['victim_position_x', 'victim_position_y']].values, 

                              death_final_circle_erg[['killer_position_x', 'killer_position_y']].values])*4096/800000

final_circle_mrm = np.vstack([death_final_circle_mrm[['victim_position_x', 'victim_position_y']].values,

                              death_final_circle_mrm[['killer_position_x', 'killer_position_y']].values])*1000/800000
bg = imread('../input/erangel.jpg')

hmap, extent = heatmap(final_circle_erg[:,0], final_circle_erg[:,1], 1.5)

alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)

colors = Normalize(0, hmap.max(), clip=True)(hmap)

colors = cm.Reds(colors)

colors[..., -1] = alphas



fig, ax = plt.subplots(figsize=(24,24))

ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)

ax.imshow(bg)

ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)

#plt.scatter(plot_data_er[:,0], plot_data_er[:,1])

plt.gca().invert_yaxis()
bg = imread('../input/miramar.jpg')

hmap, extent = heatmap(final_circle_mrm[:,0], final_circle_mrm[:,1], 1.5)

alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)

colors = Normalize(0, hmap.max(), clip=True)(hmap)

colors = cm.Reds(colors)

colors[..., -1] = alphas



fig, ax = plt.subplots(figsize=(24,24))

ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)

ax.imshow(bg)

ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)

#plt.scatter(plot_data_mr[:,0], plot_data_mr[:,1])

plt.gca().invert_yaxis()
erg_died_of = dfd.loc[(dfd['map']=='ERANGEL')&(dfd['killer_position_x']>0)&(dfd['victim_position_x']>0)&(dfd['killed_by']!='Down and Out'),:]

mrm_died_of = dfd.loc[(dfd['map']=='MIRAMAR')&(dfd['killer_position_x']>0)&(dfd['victim_position_x']>0)&(dfd['killed_by']!='Down and Out'),:]
erg_died_of['killed_by'].value_counts()[:10].plot.barh(figsize=(10,5))

plt.xlabel("Be_killed", fontsize=12)

plt.ylabel("weapon", fontsize=12)

plt.title('Weapon VS Be_killed - Island', fontsize=12)

plt.yticks(fontsize=12)
mrm_died_of['killed_by'].value_counts()[:10].plot.barh(figsize=(10,5))

plt.xlabel("Be_killed", fontsize=12)

plt.ylabel("weapon", fontsize=12)

plt.title('Weapon VS Be_killed - Erg', fontsize=12)

plt.yticks(fontsize=12)
erg_distance = np.sqrt(((erg_died_of['killer_position_x']-erg_died_of['victim_position_x'])/100)**2 + ((erg_died_of['killer_position_y']-erg_died_of['victim_position_y'])/100)**2)

mrm_distance = np.sqrt(((mrm_died_of['killer_position_x']-mrm_died_of['victim_position_x'])/100)**2 + ((mrm_died_of['killer_position_y']-mrm_died_of['victim_position_y'])/100)**2)
sns.distplot(erg_distance.loc[erg_distance<400])
erg_died_of['erg_dist'] = erg_distance

erg_died_of = erg_died_of.loc[erg_died_of['erg_dist']<800, :]

top_weapons_erg = list(erg_died_of['killed_by'].value_counts()[:10].index)

top_weapon_kills = erg_died_of[np.in1d(erg_died_of['killed_by'], top_weapons_erg)].copy()

top_weapon_kills['bin'] = pd.cut(top_weapon_kills['erg_dist'], np.arange(0, 800, 10), include_lowest=True, labels=False)

top_weapon_kills_wide = top_weapon_kills.groupby(['killed_by', 'bin']).size().unstack(fill_value=0).transpose()
top_weapon_kills_wide = top_weapon_kills_wide.div(top_weapon_kills_wide.sum(axis=1), axis=0)
from bokeh.models.tools import HoverTool

from bokeh.palettes import brewer

from bokeh.plotting import figure, show, output_notebook

from bokeh.models.sources import ColumnDataSource
def stacked(df):

    df_top = df.cumsum(axis=1)

    df_bottom = df_top.shift(axis=1).fillna(0)[::-1]

    df_stack = pd.concat([df_bottom, df_top], ignore_index=True)

    return df_stack



hover = HoverTool(

    tooltips=[

            ("index", "$index"),

            ("weapon", "@weapon"),

            ("(x,y)", "($x, $y)")

        ],

    point_policy='follow_mouse'

    )



areas = stacked(top_weapon_kills_wide)



colors = brewer['Spectral'][areas.shape[1]]

x2 = np.hstack((top_weapon_kills_wide.index[::-1],

                top_weapon_kills_wide.index)) /0.095



TOOLS="pan,wheel_zoom,box_zoom,reset,previewsave"

output_notebook()

p = figure(x_range=(1, 800), y_range=(0, 1), tools=[TOOLS, hover], plot_width=800)

p.grid.minor_grid_line_color = '#eeeeee'
source = ColumnDataSource(data={

    'x': [x2] * areas.shape[1],

    'y': [areas[c].values for c in areas],

    'weapon': list(top_weapon_kills_wide.columns),

    'color': colors

})



p.patches('x', 'y', source=source, legend="weapon",

          color='color', alpha=0.8, line_color=None)

p.title.text = "Top10 weapon kill % in diff distance - Isl"

p.xaxis.axis_label = "kill distance 0-800m"

p.yaxis.axis_label = "%"



show(p)
mrm_died_of['erg_dist'] = mrm_distance

mrm_died_of = mrm_died_of.loc[mrm_died_of['erg_dist']<800, :]

top_weapons_erg = list(mrm_died_of['killed_by'].value_counts()[:10].index)

top_weapon_kills = mrm_died_of[np.in1d(mrm_died_of['killed_by'], top_weapons_erg)].copy()

top_weapon_kills['bin'] = pd.cut(top_weapon_kills['erg_dist'], np.arange(0, 800, 10), include_lowest=True, labels=False)

top_weapon_kills_wide = top_weapon_kills.groupby(['killed_by', 'bin']).size().unstack(fill_value=0).transpose()
source = ColumnDataSource(data={

    'x': [x2] * areas.shape[1],

    'y': [areas[c].values for c in areas],

    'weapon': list(top_weapon_kills_wide.columns),

    'color': colors

})



p.patches('x', 'y', source=source, legend="weapon",

          color='color', alpha=0.8, line_color=None)

p.title.text = "Top10 weapon kill % in diff distance - Erg"

p.xaxis.axis_label = "kill distance 0-800m"

p.yaxis.axis_label = "%"



show(p)