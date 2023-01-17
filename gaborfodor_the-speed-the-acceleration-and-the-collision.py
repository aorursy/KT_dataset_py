%matplotlib inline
import os
import pandas as pd
import datetime as dt
import numpy as np
from tqdm import tqdm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from IPython.display import HTML
from matplotlib import animation, rc
import matplotlib.image as mpimg
import warnings

def write_image(fig, filename, save=False):
    if save:
        try:
            import plotly.io as pio
            pio.write_image(fig, './svgs/' + filename)
        except Exception:
            pass
C = ['#3D0553', '#4D798C', '#7DC170', '#F7E642']
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
pd.set_option('display.max_columns', 99)
start = dt.datetime.now()

NFL_DATA_DIR = '../input/NFL-Punt-Analytics-Competition'
ALL_PLAYS_PATH = '../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv'
NGS_DIR = '../input/next-gen-stats-by-play'
EDA_DIR = '../input/exploratory-data-analysis-external-data'
EXT_DIR = '../input/externalnfl/'

YARD = 0.9144
MPH = 1.609344
SPEED_LIMIT = 13
MAX_SPEED = 11
SMOOTHING_FACTOR = 3
ACCELERATION_BINS = [0, 10, 15, 100]

PUNT_COVERAGE_ROLES = [
    'GL', 'GR', 'P', 'PPL', 'PPR', 'PC', 'PLW', 'PRW', 'PLT', 'PLG', 'PLS', 'PRG', 'PRT']
PUNT_RETURN_ROLES = ['PR', 'PFB', 'PLL', 'PDM', 'PLM', 'PLR', 'VL', 'VR', 'PDL', 'PDR']

def get_plays():
    data = pd.read_csv(os.path.join(NFL_DATA_DIR, 'play_information.csv'),
                       parse_dates=['Game_Date'])
    data.columns = [col.replace('_', '') for col in data.columns]
    data['PlayKey'] = data['GameKey'].apply(str) + '_' + data['PlayID'].apply(str)
    data = data.drop(['PlayID', 'PlayType'], axis=1)
    data = data.sort_values(['GameKey', 'Quarter', 'GameClock'])
    data['PlayType'] = 'Punt'
    return data

def get_ngs(playkey):
    ngs = pd.read_csv(os.path.join(NGS_DIR, f'ngs_{playkey}.csv'), parse_dates=['Time'])
    ngs['t'] = (ngs.Time - ngs.Time.min()) / np.timedelta64(1, 's')
    ngs = ngs.sort_values(by='t')
    return ngs

def get_punt_players():
    player_role = pd.read_csv(os.path.join(NFL_DATA_DIR, 'play_player_role_data.csv'))
    player_role.columns = [col.replace('_', '') for col in player_role.columns]
    player_role['PlayKey'] = player_role['GameKey'].apply(str) + '_' + player_role[
        'PlayID'].apply(str)
    player_role['ShortRole'] = player_role['Role'].apply(
        lambda s: s.replace('i', '').replace('o', '')[:3])
    player_role['PuntCoverage'] = player_role['ShortRole'].apply(
        lambda s: s in PUNT_COVERAGE_ROLES)
    player_role['PuntReturn'] = player_role['ShortRole'].apply(lambda s: s in PUNT_RETURN_ROLES)

    players = get_players()

    return player_role.merge(players, how='left', on='GSISID')

def get_players():
    players = pd.read_csv(os.path.join(NFL_DATA_DIR, 'player_punt_data.csv'))
    players = players.groupby('GSISID').agg({
        'Number': lambda x: ','.join(
            x.replace(to_replace='[^0-9]', value='', regex=True).unique()),
        'Position': lambda x: ','.join(x.unique())})
    return players.reset_index()

def get_punt_player_speed():
    player_punt_speed = pd.read_csv(os.path.join(NGS_DIR, 'player_ngs.csv'))
    punt_player_speed = pd.merge(player_punt_speed, get_punt_players(),
                                 on=['PlayKey', 'GSISID'])
    punt_player_speed = punt_player_speed[punt_player_speed.MaxSpeed < SPEED_LIMIT]
    punt_player_speed['CollisionId'] = np.digitize(-punt_player_speed.MinAcceleration,
                                                   bins=ACCELERATION_BINS)
    punt_player_speed['Collision'] = punt_player_speed['CollisionId'].replace(
        {1: 'Mild', 2: 'Medium', 3: 'Serious'})
    return punt_player_speed

def get_video_review():
    data = pd.read_csv(os.path.join(NFL_DATA_DIR, 'video_review.csv'))
    data.columns = [col.replace('_', '') for col in data.columns]
    data['PlayKey'] = data['GameKey'].apply(str) + '_' + data['PlayID'].apply(str)

    footage = pd.read_csv(os.path.join(NFL_DATA_DIR, 'video_footage-injury.csv'))
    footage['PlayKey'] = footage['gamekey'].apply(str) + '_' + footage['playid'].apply(str)

    footage = footage.rename(columns={'PREVIEW LINK (5000K)': 'VideoLink'})
    data = data.merge(footage[['PlayKey', 'VideoLink', 'PlayDescription']],
                      how='left',
                      on=['PlayKey'])
    data['PrimaryPartnerGSISID'] = data['PrimaryPartnerGSISID'].replace('Unclear', np.nan)
    data = data.fillna({'PrimaryPartnerGSISID': -999})
    data['PrimaryPartnerGSISID'] = data['PrimaryPartnerGSISID'].astype('int64')
    return data

def calculate_speed_and_acceleration(ngs, smoothing_factor=5):
    speed = ngs.pivot('t', 'GSISID', 'dis') * YARD
    speed = speed.fillna(0)
    speed = speed.rolling(smoothing_factor).mean() * 10
    acc = speed.clip(0, MAX_SPEED).diff(smoothing_factor) * 10. / smoothing_factor
    return speed, acc

def collect_ngs_player_stats():
    plays = get_plays()
    result = []
    for playkey in tqdm(plays.PlayKey.values):
        try:
            ngs = get_ngs(playkey)

            speed, acc = calculate_speed_and_acceleration(ngs, SMOOTHING_FACTOR)
            max_speed = speed.max(axis=0).reset_index().rename(columns={0: 'MaxSpeed'})
            min_acceleration = acc.min(axis=0).reset_index().rename(
                columns={0: 'MinAcceleration'})

            collision_coords = pd.DataFrame([[c, acc[c].argmin()] for c in acc.columns],
                                            columns=['GSISID', 't'])
            collision_coords = collision_coords.merge(ngs[['GSISID', 't', 'x', 'y']],
                                                      how='left', on=['GSISID', 't'])
            collision_coords['x'] = collision_coords['x'] - 10
            collision_coords.columns = ['GSISID', 'CollisionTime', 'CollisionX', 'CollisionY']

            stats = pd.merge(max_speed, min_acceleration, on='GSISID')
            stats = stats.merge(collision_coords, on='GSISID', how='left')
            stats['PlayKey'] = playkey
            result.append(stats)
        except Exception as e:
            print(e)
    return pd.concat(result)

def show_injured_player_speed_profile(playkey, a, b, smoothing_factor=5):
    speed, acc = calculate_speed_and_acceleration(get_ngs(playkey), smoothing_factor)
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(speed[a], color=C[0], lw=3, alpha=0.8, label='Injured Player')
    axs[0].plot(speed.mean(axis=1), color=C[-1], lw=2, alpha=0.5, label='All Player Average')
    axs[0].set_ylabel('Speed (m/s)')
    axs[1].set_ylabel('Acceleration (m/s2)')
    axs[1].plot(acc[a], color=C[0], lw=3, alpha=0.8)
    try:
        axs[0].plot(speed[int(b)], color=C[1], lw=3, alpha=0.8, label='Primary Partner')
        axs[1].plot(acc[int(b)], color=C[1], lw=3, alpha=0.8)
    except Exception as e:
        print(e)
    plt.xlabel('Time (s)')
    axs[0].grid()
    axs[1].grid()
    axs[0].legend(loc=0)
    axs[0].set_ylim(0, 10)
    axs[1].set_ylim(-15, 10)
    plt.show()
    fig.savefig(f'speed_profile_{playkey}.png', dpi=300)

player_ngs = pd.read_csv(os.path.join(NGS_DIR, 'player_ngs.csv'))
print(player_ngs.shape)
print(player_ngs.count())
playkey = '274_3609'
ngs = get_ngs(playkey)
ngs.shape
ngs.head()
speed, acc = calculate_speed_and_acceleration(ngs, smoothing_factor=5)
show_injured_player_speed_profile(playkey, 23742, 31785, 3)
punt_player_speed = get_punt_player_speed()
punt_player_speed.describe()
punt_player_speed = get_punt_player_speed()
punt_player_speed.shape
punt_player_speed.head()

video_info = get_video_review()
video_info = video_info.merge(
    punt_player_speed[['GSISID', 'PlayKey', 'MaxSpeed', 'MinAcceleration',
                       'CollisionTime', 'CollisionX', 'CollisionY']],
    on=['GSISID', 'PlayKey'],
    how='left')
video_info = video_info.merge(
    punt_player_speed[['GSISID', 'PlayKey', 'MaxSpeed', 'MinAcceleration',
                       'CollisionTime', 'CollisionX', 'CollisionY']],
    left_on=['PrimaryPartnerGSISID', 'PlayKey'],
    right_on=['GSISID', 'PlayKey'],
    how='left',
    suffixes=['', 'PrimaryPartner'])
video_info['MaxMaxPSpeed'] = video_info[['MaxSpeed', 'MaxSpeedPrimaryPartner']].max(axis=1)
video_info['MinMinAcc'] = video_info[
    ['MinAcceleration', 'MinAccelerationPrimaryPartner']].min(axis=1)
video_info.shape
video_info.head()
video_info.to_csv('video_info_collision.csv', index=False)
video_info['Concussion'] = 1

fig, ax = plt.subplots()
sns.distplot(punt_player_speed.MaxSpeed, bins=20, kde_kws=dict(shade=True),
             kde=True, color=C[1], ax=ax, label='MaxSpeed')
plt.plot(video_info['MaxMaxPSpeed'].values, 0.01 * np.ones(len(video_info)),
         'kx', alpha=0.8, markersize=10, lw=3, label='Injured player or partner max speed')
plt.xlim(0, 12)
plt.xticks(range(0, 12, 1))
plt.legend(loc=0)
plt.ylabel('Probability Density')
plt.xlabel('Max Speed (m/s)')
plt.title('Player Max Speed Distribution')
plt.grid()
plt.show();

fig, ax = plt.subplots()
sns.distplot(punt_player_speed.MinAcceleration, bins=20, kde_kws=dict(shade=True),
             kde=True, color=C[0], ax=ax, label='MinAcceleration')
plt.plot(video_info['MinMinAcc'].values, 0.01 * np.ones(len(video_info)),
         'kx', alpha=0.8, markersize=10, lw=3, label='Injured player or partner acceleration')
plt.legend(loc=0)
plt.ylabel('Probability Density')
plt.xlabel('Min Acceleration (m/s2)')
plt.title('Player Acceleration Distribution')
plt.grid()
plt.show();

play_acceleration = punt_player_speed.groupby('PlayKey')[['MinAcceleration']].min()
play_acceleration = play_acceleration.reset_index()
play_acceleration = play_acceleration.merge(video_info[['PlayKey', 'Concussion']],
                                            on='PlayKey',
                                            how='left')
play_acceleratiaon = play_acceleration.fillna(0)
play_acceleration['WorstCollisionId'] = np.digitize(-play_acceleration.MinAcceleration,
                                                    bins=ACCELERATION_BINS)
play_acceleration['WorstCollision'] = play_acceleration['WorstCollisionId'].replace(
    {1: 'Mild', 2: 'Medium', 3: 'Serious'})

r = play_acceleration.groupby('WorstCollision')[['Concussion', 'MinAcceleration']].mean()
p = play_acceleration.groupby('WorstCollision')[['Concussion']].count()
c = play_acceleration.groupby('WorstCollision')[['Concussion']].sum()
collisions = pd.concat([r, p, c], axis=1)
collisions.columns =['ConcussionRate', 'MinAcceleration', '#PuntPlays', '#Concussions']
collisions.sort_values(by='MinAcceleration')
punts = pd.read_csv(os.path.join(EDA_DIR, 'punts.csv'))
punts.shape
punts.head()
punt_player_speed['SeriousCollision'] = 1 * punt_player_speed['Collision'] == 'Serious'
punt_player_speed['MediumCollision'] = 1 * punt_player_speed['Collision'] == 'Medium'
play_collisions = punt_player_speed.groupby('PlayKey')[
    ['SeriousCollision', 'MediumCollision']].sum()
play_collisions = play_collisions.merge(punts, on='PlayKey')

c = play_collisions.groupby('PuntType')[['GameDate']].count()
s = play_collisions.groupby('PuntType')[['SeriousCollision', 'MediumCollision']].sum()
m = play_collisions.groupby('PuntType')[['SeriousCollision', 'MediumCollision']].mean()

punt_type_collisions = pd.concat([c, s, m], axis=1)
punt_type_collisions.columns = ['#Plays', '#SeriousCollision', '#MediumCollision',
                                'SeriousCollision', 'MediumCollision']
punt_type_collisions['#TotalCollisions'] = punt_type_collisions['#SeriousCollision'] + \
                                           punt_type_collisions['#MediumCollision']
punt_type_collisions['TotalCollisions'] = punt_type_collisions['SeriousCollision'] + \
                                          punt_type_collisions['MediumCollision']
punt_type_collisions = punt_type_collisions.sort_values(by='TotalCollisions', ascending=False)
punt_type_collisions
punt_type_collisions = punt_type_collisions[
    ~punt_type_collisions.index.isin(['OTHER', 'NOPLAY'])]
punt_type_collisions.sum()

data = [
    go.Bar(
        y=punt_type_collisions['MediumCollision'].values,
        x=punt_type_collisions.index.values,
        marker=dict(color=C[1]),
        text=punt_type_collisions.index.values,
        name='Medium'
    ),
    go.Bar(
        y=punt_type_collisions['SeriousCollision'].values,
        x=punt_type_collisions.index.values,
        marker=dict(color=C[0]),
        text=punt_type_collisions.index.values,
        name='Serious'
    ),
]
layout = go.Layout(
    title='Collisions per  punt play',
    barmode='stack',
    hovermode='closest',
    xaxis=dict(title='Punt Type', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Average dangerous collision per punt', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PuntTypeCollisionBar')

data = [
    go.Scatter(
        y=punt_type_collisions['TotalCollisions'].values,
        x=punt_type_collisions.index.values,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(punt_type_collisions['#Plays'].values),
                    color=punt_type_collisions['TotalCollisions'].values,
                    colorscale='Viridis',
                    reversescale=True,
                    showscale=True
                    ),
        text=punt_type_collisions['#Plays'].values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Collisions per punt play',
    hovermode='closest',
    xaxis=dict(title='Punt Type', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Average dangerous collisions per punt', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='WeeklyTopTopics')
write_image(fig, 'CollisionScatter.svg')

player_collisions = punt_player_speed.groupby(['ShortRole', 'PuntReturn'])[
    ['SeriousCollision', 'MediumCollision']].mean()
player_collisions['TotalCollisions'] = player_collisions['SeriousCollision'] + \
                                       player_collisions['MediumCollision']
player_collisions = player_collisions.reset_index()
player_collisions = player_collisions.sort_values(by=[
    'PuntReturn', 'TotalCollisions'], ascending=False)
player_collisions['color'] = player_collisions.PuntReturn.replace({True: C[1], False: C[0]})
player_collisions

data = [
    go.Bar(
        y=player_collisions['TotalCollisions'].values,
        x=player_collisions.ShortRole.values,
        marker=dict(color=player_collisions.color.values),
        text=player_collisions.ShortRole.values,
        name='Medium'
    )]
layout = go.Layout(
    title='Player Collisions',
    barmode='stack',
    hovermode='closest',
    xaxis=dict(title='Punt Return and Punt Coverge Roles',
               ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Probablity of collision for each player', ticklen=5, gridwidth=2),
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PlayerRoleCollisionBar')
write_image(fig, 'PlayerRoleCollisionBar.svg')

collision_coords = punt_player_speed[punt_player_speed.Collision.isin(['Medium', 'Serious'])]
collision_coords = collision_coords[['CollisionX', 'CollisionY']].dropna()
collision_coords['CollisionX'] = collision_coords['CollisionX'].clip(0, 100)
collision_coords['CollisionY'] = collision_coords['CollisionY'].clip(0, 54)
collision_coords.shape
collision_coords.head(2)
collision_coords.describe()
g = sns.jointplot("CollisionX", "CollisionY", data=collision_coords,
                  kind="kde", space=0, color=C[1])
g.ax_joint.plot(collision_coords.CollisionX, collision_coords.CollisionY, 'k+', alpha=0.1)
g.ax_joint.set_xlim(0, 100)
g.ax_joint.set_xticks(range(0, 101, 10))
g.ax_joint.set_xticklabels(
    ['GL', '10', '20', '30', '40', '50', '40', '30', '20', '10', 'GL'])
g.ax_joint.set_ylim(-7, 60)
g.ax_joint.set_yticks([0, 53.3])
g.ax_joint.set_yticklabels(['Sideline', 'Sideline'])
g.ax_joint.grid()
g.fig.set_figheight(8)
g.fig.set_figwidth(16)
g.fig.savefig('CollisionCoords.svg')
plt.show();

playkey = '397_1526'
ngs = get_ngs(playkey)
ngs['PlayKey'] = playkey
ngs['x'] = ngs['x'] - 10
ngs['px'] = 244 + ngs['x'] / 100 * (1920 - 488)
ngs['py'] = 100 + (53.3 - ngs['y']) / 53.3 * (957 - 200)
p = punt_player_speed[['PlayKey', 'GSISID', 'ShortRole', 'PuntReturn',
                       'CollisionX', 'CollisionY', 'MinAcceleration', 'Collision']]
ngs = ngs.merge(p, on=['PlayKey', 'GSISID'])
ngs = ngs[(ngs.t >= 8) & (ngs.t <= 17)]

rc('animation', html='jshtml', embed_limit=100)
fig, c_ax = plt.subplots(1, 1, figsize=(20, 10))
field = mpimg.imread(os.path.join(EXT_DIR, 'field home1920.png'))
c_ax.imshow(field)
c_ax.axis('off')
step_count = 30
step_length = 10 * 1000 / step_count
time_steps = np.linspace(ngs['t'].min(), ngs['t'].max(), step_count + 1)

def update_frame(i):
    n_rows = ngs[ngs['t'] <= time_steps[i + 1]]
    n_rows = n_rows[n_rows['t'] > time_steps[i]]
    for (player_id, punt_return_team, role), player_rows in n_rows.groupby(
            ['GSISID', 'PuntReturn', 'ShortRole']):
        player_rows = player_rows.sort_values('t')
        color = C[0] if punt_return_team else C[-1]
        alpha = 1. if player_id in (32894, 31763) else 0.2
        c_ax.plot(player_rows['px'], player_rows['py'], color=color, alpha=alpha,
                  label='_nolegend_', lw=2)

a = animation.FuncAnimation(fig, update_frame, range(step_count), interval=step_length)
plt.close();

HTML(a.to_jshtml())

HTML('''<video width="800" height="450" controls>
  <source src="https://s3-eu-west-1.amazonaws.com/nfl-punt-analytics/BudapesPythonsFromRawDataToInsights.mp4" type="video/mp4">
Your browser does not support the video tag.</video>''')

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))