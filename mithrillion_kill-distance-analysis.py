import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib import style
from scipy.misc.pilutil import imread
plt.rcParams['figure.figsize'] = (10, 6)
style.use('ggplot')
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import brewer
from bokeh.models.tools import HoverTool
from bokeh.models.sources import ColumnDataSource

from collections import Counter
sample = pd.read_csv("../input/pubg-match-deaths/deaths/kill_match_stats_final_0.csv")
sample.shape
# small = sample.loc[(sample['map'] == 'MIRAMAR') & (sample['killer_position_x'] != 0)
#                   & (sample['victim_position_x'] != 0)][:10000].copy()
small = sample.loc[(sample['map'] == 'MIRAMAR')][:10000].copy()

x_diffs = small['killer_position_x'] - small['victim_position_x']
y_diffs = small['killer_position_y'] - small['victim_position_y']
sq_diffs = x_diffs ** 2 + y_diffs ** 2
dists = np.sqrt(sq_diffs)
log_dists = np.log10(1 + dists)
small.head()
sns.distplot(log_dists.dropna());
small.loc[log_dists < 1, :].groupby('killed_by').size().plot.bar();
small.loc[log_dists > 5, :].groupby('killed_by').size().plot.bar();
bg = imread("../input/high-res-map-miramar/miramar-large.png")
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(bg);
def kill_viz(killer_pos, victim_pos, bg, bg_width, bg_height, zoom=False, figsize=(15, 15)):
    fig, ax = plt.subplots(figsize=figsize)
    x, y = killer_pos
    tx, ty = victim_pos
    dx, dy = tx - x, ty - y
    x *= bg_width / 800000
    y *= bg_height / 800000
    dx *= bg_width / 800000
    dy *= bg_height / 800000
    ax.imshow(bg)
    arrow_width = min((abs(dx) + abs(dy)), (bg_width + bg_height) * 0.2) * 0.02
    if zoom:
        edge = max(abs(dx), abs(dy))
        ax.set_xlim(max(0, min(x, x + dx) - edge * 5), min(max(x, x + dx) + edge * 5, bg_width))
        ax.set_ylim(min(max(y, y + dy) + edge * 5, bg_width), max(min(y, y + dy) - edge * 5, 0))
    ax.arrow(x, y, dx, dy, width=arrow_width, color='r', length_includes_head=True)
    plt.show()
temp = small.loc[(log_dists > 5)].iloc[0]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)
kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(10, 10))
temp = small.loc[(log_dists > 5)].iloc[50]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)
kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(10, 10))
temp = small[small['killer_position_x'] == 0].loc[(log_dists > 5)].iloc[0]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)
kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(10, 10))
small = sample.loc[(sample['map'] == 'MIRAMAR') & (sample['killer_position_x'] != 0)
                  & (sample['victim_position_x'] != 0)][:10000].copy()

x_diffs = small['killer_position_x'] - small['victim_position_x']
y_diffs = small['killer_position_y'] - small['victim_position_y']
sq_diffs = x_diffs ** 2 + y_diffs ** 2
dists = np.sqrt(sq_diffs)
log_dists = np.log10(1 + dists)
small['log_dist'] = log_dists
sns.distplot(log_dists.dropna());
small.loc[log_dists > 4.5, :].groupby('killed_by').size().plot.bar();
temp = small.loc[(log_dists == np.max(log_dists))].iloc[0]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)
kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(12, 12))
top_weapons = list(small[small['killed_by'] != 'Down and Out'].groupby('killed_by').size()\
                     .sort_values(ascending=False)[:10].index)
top_weapon_kills = small[np.in1d(small['killed_by'], top_weapons)].copy()
top_weapon_kills['bin'] = pd.cut(top_weapon_kills['log_dist'], np.arange(0, 6.2, 0.2), include_lowest=True, labels=False)
top_weapon_kills_wide = top_weapon_kills.groupby(['killed_by', 'bin']).size().unstack(fill_value=0).transpose()
def  stacked(df):
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
                top_weapon_kills_wide.index)) / 5

TOOLS="pan,wheel_zoom,box_zoom,reset,previewsave"
output_notebook()
p = figure(x_range=(1, 5), y_range=(0, 800), tools=[TOOLS, hover], plot_width=800)
p.grid.minor_grid_line_color = '#eeeeee'

source = ColumnDataSource(data={
    'x': [x2] * areas.shape[1],
    'y': [areas[c].values for c in areas],
    'weapon': list(top_weapon_kills_wide.columns),
    'color': colors
})

p.patches('x', 'y', source=source, legend="weapon",
          color='color', alpha=0.8, line_color=None)
p.title.text = "Distribution of Kill Distance per Weapon"
p.xaxis.axis_label = "log10 of kill distance"
p.yaxis.axis_label = "number of kills"

show(p)
