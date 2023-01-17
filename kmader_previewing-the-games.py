%matplotlib inline
import os
import pandas as pd
import numpy as np
import glob
from plotly import offline
import feather
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
football_field = lambda : Rectangle(xy=(10, 0), 
                                    width=100, 
                                    height=53.3, 
                                    color='g',
                                   alpha=0.10)
plt.style.use('ggplot')
pd.set_option('max.columns', None)
offline.init_notebook_mode()
feather_dir = '../input/convert-to-feather-for-use-in-other-kernels/'
nfl_dir = '../input/NFL-Punt-Analytics-Competition/'
# Load in a sample 
all_games_df = pd.read_feather(
    os.path.join(feather_dir, 'ngs.feather')
)
all_games_df.sample(5)
fig, m_axs = plt.subplots(8, 8, figsize=(20, 20))
for (play_id, play_rows), c_ax in zip(all_games_df.groupby('PlayID'), 
                      m_axs.flatten()):
    c_ax.add_patch(football_field())
    for player_id, player_rows in play_rows.groupby('GSISID'):
        player_rows = player_rows.sort_values('Time')
        c_ax.plot(player_rows['x'], player_rows['y'], 
                  label=player_id)
    c_ax.set_title(play_id)
    c_ax.set_aspect(1)
    c_ax.set_xlim(0, 120)
    c_ax.set_ylim(-10, 63)
match_cols = ['Season_Year', 'GameKey', 'PlayID']
video_review_df = pd.read_csv(os.path.join(nfl_dir, 
                                           'video_review.csv'))
video_review_df.dropna(subset=['GSISID'], inplace=True)
video_review_df['GSISID'] = video_review_df['GSISID'].map(int)

for c_col in match_cols:
    # match types to make merges later quicker
    video_review_df[c_col] = video_review_df[c_col].astype(all_games_df[c_col].dtype)
video_review_df.head(5)
def player_scene_key(in_df):
    return in_df.apply(lambda x: (x['Season_Year'],
                                x['GameKey'],
                                x['PlayID'],
                                x['GSISID']
                               ), 1)
conc_keys = player_scene_key(video_review_df).values
conc_dict = dict(
    zip(conc_keys, 
        video_review_df['Player_Activity_Derived'].values)
)
sample_plays = 5
fig, m_axs = plt.subplots(4, sample_plays, figsize=(20, 20))

for (conc_type, conc_rows), n_axs in zip(
    video_review_df.groupby('Player_Activity_Derived'),
    m_axs
):
    sel_plays = conc_rows.sample(sample_plays)
    
    for (play_id, c_play_row), c_ax in zip(
        sel_plays.groupby('PlayID'), 
        n_axs):
        play_rows = pd.merge(c_play_row[match_cols], 
                             all_games_df)
        c_ax.add_patch(football_field())
        for player_id, player_rows in play_rows.groupby('GSISID'):
            player_rows = player_rows.sort_values('Time')
            if player_id in c_play_row['GSISID'].values:
                c_ax.plot(player_rows['x'], player_rows['y'], 
                          'r.-', label='Primary')
            elif player_id in c_play_row['Primary_Partner_GSISID'].values:
                c_ax.plot(player_rows['x'], player_rows['y'], 
                          'b.-', label='Partner')
            else:
                c_ax.plot(player_rows['x'], player_rows['y'], 
                          alpha=0.5, label='_nolegend_')
                
        c_ax.set_title(play_id)
        c_ax.set_aspect(1)
        c_ax.set_xlim(0, 120)
        c_ax.set_ylim(-5, 68)
        c_ax.legend()
    n_axs[0].set_title('{0}-#{1} plays'.format(conc_type, conc_rows.shape[0]))
fig, c_ax = plt.subplots(1, 1, figsize=(20, 10))
q_rows = play_rows.copy()
dis_scalar = 1/10.0
q_rows['u'] = -1*q_rows['dis']/dis_scalar*np.sin(q_rows['o']*2*np.pi/360)
q_rows['v'] = q_rows['dis']/dis_scalar*np.cos(q_rows['o']*2*np.pi/360)
c_ax.add_patch(football_field())
for player_id, player_rows in q_rows.groupby('GSISID'):
    player_rows = player_rows.sort_values('Time')
    if player_id in c_play_row['GSISID'].values:
        c_ax.quiver(player_rows['x'], player_rows['y'],
                    player_rows['u'], player_rows['v'],
                    units='x', label='Primary')
    else:
        c_ax.plot(player_rows['x'], player_rows['y'], 
                  alpha=0.5, label='_nolegend_')
c_ax.set_title(play_id)
c_ax.set_aspect(1)
c_ax.set_xlim(0, 120)
c_ax.set_ylim(-10, 63)
from matplotlib import animation, rc
rc('animation', html='jshtml', embed_limit=100)
fig, c_ax = plt.subplots(1, 1, figsize=(20, 10))
c_ax.add_patch(football_field())
c_ax.set_aspect(1)
c_ax.set_xlim(0, 120)
c_ax.set_ylim(-10, 63)
q_rows['clock'] = (q_rows['Time']-q_rows['Time'].min()).dt.total_seconds()

step_count = 30
step_length = 10*1000/(step_count)
time_steps = np.linspace(q_rows['clock'].min(),
                       q_rows['clock'].max(),
                       step_count+1)
def update_frame(i):
    n_rows = q_rows[q_rows['clock']<=time_steps[i+1]]
    n_rows = n_rows[n_rows['clock']>time_steps[i]]
        
    for player_id, player_rows in n_rows.groupby('GSISID'):
        player_rows = player_rows.sort_values('Time')
        if player_id in c_play_row['GSISID'].values:
            c_ax.quiver(player_rows['x'], player_rows['y'],
                        player_rows['u'], player_rows['v'],
                        units='x', label='Primary')
        else:
            c_ax.plot(player_rows['x'], player_rows['y'], 
                      alpha=0.5, label='_nolegend_')
ani = animation.FuncAnimation(fig, 
                              update_frame, 
                              range(step_count), 
                              interval=step_length)
ani
