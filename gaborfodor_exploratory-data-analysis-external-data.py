%matplotlib inline
import os
import re
import pandas as pd
import datetime as dt
import numpy as np
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
DATASET_DIR = '../dataset'

def get_games():
    data = pd.read_csv(os.path.join(NFL_DATA_DIR, 'game_data.csv'), parse_dates=['Game_Date'])
    data.columns = [col.replace('_', '') for col in data.columns]
    data = data.drop(
        ['StadiumType', 'Turf', 'GameWeather', 'Temperature', 'OutdoorWeather', 'Stadium'],
        axis=1)
    return data

def get_teams(games):
    home = games[['HomeTeamCode', 'HomeTeam']]
    visit = games[['HomeTeamCode', 'HomeTeam']]
    home.columns = ['TeamCode', 'Team']
    visit.columns = ['TeamCode', 'Team']
    return pd.concat([home, visit]).drop_duplicates()

def get_plays():
    data = pd.read_csv(os.path.join(NFL_DATA_DIR, 'play_information.csv'),
                       parse_dates=['Game_Date'])
    data.columns = [col.replace('_', '') for col in data.columns]
    data['PlayKey'] = data['GameKey'].apply(str) + '_' + data['PlayID'].apply(str)
    data = data.drop(['PlayID', 'PlayType'], axis=1)
    data = data.sort_values(['GameKey', 'Quarter', 'GameClock'])
    data['PlayType'] = 'Punt'
    return data

games = get_games()
plays = get_plays()
teams = get_teams(games)
'Teams:, {} Games: {}, Punt Plays: {}'.format(teams.shape, games.shape, plays.shape)
h = games.groupby('HomeTeam')[['GameKey']].count().sort_values(by='GameKey').rename(
    columns={'GameKey': 'Games'})
h.head()

games.head(2)
plays.head(2)
games[games.HomeTeam.isin(['AFC', 'NFC'])]
plays[plays['GameKey'].isin([333, 666])]
plays.groupby(['SeasonYear', 'SeasonType'])[['PlayKey']].count()
for desc in plays.PlayDescription.values[:4]:
    print(desc)
class PuntType:
    TOUCHBACK = 'TOUCHBACK'
    FAIRCATCH = 'FAIRCATCH'
    MUFFS = 'MUFFS'
    RETURN = 'RETURN'
    OUTOFBOUNDS = 'OUTOFBOUNDS'
    NOPLAY = 'NOPLAY'
    DOWNED = 'DOWNED'
    OTHER = 'OTHER'


PUNT_COVERAGE_ROLES = ['GL', 'GR', 'P', 'PPL', 'PPR', 'PC', 'PLW', 'PRW', 'PLT', 'PLG', 'PLS',
                       'PRG', 'PRT']
PUNT_RETURN_ROLES = ['PR', 'PFB', 'PLL', 'PDM', 'PLM', 'PLR', 'VL', 'VR', 'PDL', 'PDR']

PENALTIES = ['Offensive Holding', 'False Start', 'Defensive Pass Interference',
             'Defensive Holding', 'Unnecessary Roughness', 'Defensive Offside',
             'Illegal Block Above the Waist', 'Delay of Game', 'Neutral Zone Infraction',
             'Offensive Pass Interference', 'Illegal Use of Hands', 'Roughing the Passer',
             'Face Mask (15 Yards)', 'Unsportsmanlike Conduct', 'Illegal Formation',
             'Encroachment', 'Defensive 12 On-field', 'Illegal Contact',
             'Intentional Grounding', 'Illegal Shift', 'Offside on Free Kick', 'Taunting',
             'Horse Collar Tackle', 'Ineligible Downfield Pass', 'Illegal Motion', 'Leverage',
             'Illegal Forward Pass', 'Offensive 12 On-field', 'Player Out of Bounds on Punt',
             'Chop Block', 'Running Into the Kicker', 'Tripping', 'Disqualification',
             'Illegal Blindside Block', 'Roughing the Kicker', 'Illegal Substitution',
             'Illegal Touch Pass', 'Interference with Opportunity to', 'Clipping',
             'Illegal Touch Kick', 'Fair Catch Interference', 'Ineligible Downfield Kick',
             'Low Block', 'Illegal Crackback', 'Leaping', 'Offensive Offside',
             'Defensive Delay of Game', 'Illegal Peelback', 'Invalid Fair Catch Signal',
             'False Start, superseded. PENALTY', 'Delay of Kickoff', 'Short Free Kick',
             'Illegal Bat'
             ]
SERIOUS_PENALTIES = [
    'Illegal Block Above the Waist', 'Unnecessary Roughness',
    'Unsportsmanlike Conduct', 'Face Mask (15 Yards)', 'Roughing the Kicker',
    'Illegal Blindside Block', 'Horse Collar Tackle', 'Chop Block', 'Taunting',
    'Clipping', 'Disqualification', 'Illegal Crackback'
]


class PlayDescription:
    @staticmethod
    def punts_x_yards(row):
        match_result = re.compile(".+punts ([0-9]+) yards.+").fullmatch(row['PlayDescription'])
        if match_result is not None:
            return int(match_result.groups()[0])

    @staticmethod
    def punts_from_yard(row):
        try:
            half, yard = row['YardLine'].split(' ')
            return int(yard) if half == row['PossTeam'] else 100 - int(yard)
        except Exception:
            pass

    @staticmethod
    def punt_return_distance(row):
        if 'for no gain' in row['PlayDescription']:
            return 0
        positive_return = re.compile('.+ for (-?[0-9]+) yard.+').fullmatch(
            row['PlayDescription'])
        if positive_return is not None:
            return int(positive_return.groups()[0])

    @staticmethod
    def punt_returner(row):
        match_result = re.compile(".+([A-Z]\.[A-Z][a-z]+) .*?for (-?[0-9]+) yard.*").fullmatch(
            row['PlayDescription'])
        if match_result is not None:
            player, yard = match_result.groups()
            return player

    @staticmethod
    def punt_type(row):
        desc = row['PlayDescription']
        if 'No Play' in desc:
            return PuntType.NOPLAY
        if 'touchback' in desc.lower():
            return PuntType.TOUCHBACK
        if 'fair catch' in desc.lower():
            return PuntType.FAIRCATCH
        if 'out of bounds' in desc.lower():
            return PuntType.OUTOFBOUNDS
        if 'downed by' in desc.lower():
            return PuntType.DOWNED
        if 'MUFFS' in desc:
            return PuntType.MUFFS
        if PlayDescription.punt_return_distance(row) is not None:
            return PuntType.RETURN
        else:
            return PuntType.OTHER

    @staticmethod
    def is_penalty(row):
        return 'PENALTY' in row['PlayDescription']

    @staticmethod
    def penalty_type(row):
        if PlayDescription.is_penalty(row):
            for penalty in PENALTIES:
                if penalty.lower() in row['PlayDescription'].lower():
                    return penalty
            return 'OTHER'

    @staticmethod
    def is_injury(row):
        return 'was injured during the play' in row['PlayDescription'].lower()

def apply_f_for_rows(f, df):
    return [f(row) for _, row in df.iterrows()]


def parse_play_description_field(df):
    plays = df.copy()
    plays['PuntDistance'] = apply_f_for_rows(PlayDescription.punts_x_yards, plays)
    plays['PuntStart'] = apply_f_for_rows(PlayDescription.punts_from_yard, plays)
    plays['PuntedTo'] = plays['PuntStart'] + plays['PuntDistance']
    plays['PuntType'] = apply_f_for_rows(PlayDescription.punt_type, plays)
    plays['PuntReturner'] = apply_f_for_rows(PlayDescription.punt_returner, plays)
    plays['PuntReturnedYards'] = apply_f_for_rows(PlayDescription.punt_return_distance, plays)
    plays['Penalty'] = apply_f_for_rows(PlayDescription.is_penalty, plays)
    plays['PenaltyType'] = apply_f_for_rows(PlayDescription.penalty_type, plays)
    plays['Injury'] = apply_f_for_rows(PlayDescription.is_injury, plays)
    return plays

punts = parse_play_description_field(plays)
punts.to_csv('punts.csv', index=False)
punts.head()
pc = punts[punts.SeasonType == 'Reg'].groupby('SeasonYear')[
    ['PuntDistance', 'PuntReturnedYards']].count()
pc.columns = ['PuntCount', 'PuntReturnCount']
punt_min = punts[punts.SeasonType == 'Reg'].groupby('SeasonYear')[
    ['PuntDistance', 'PuntReturnedYards']].mean()
punt_max = punts[punts.SeasonType == 'Reg'].groupby('SeasonYear')[
    ['PuntDistance', 'PuntReturnedYards']].max()
punt_stats = pd.merge(punt_min, punt_max, left_index=True, right_index=True,
                      suffixes=['Mean', 'Max'])
punt_stats = pc.merge(punt_stats, left_index=True, right_index=True)
print('Our feature statistics:')
punt_stats.reset_index()

reference_data = pd.DataFrame(
    [[2016, 2335, 1012, 45.3, 8.6, 78, 95], [2017, 2444, 1069, 45.4, 8.2, 77, 88]],
    columns=['SeasonYear', 'PuntCount', 'PuntReturnCount', 'PuntDistanceMean',
             'PuntReturnedYardsMean', 'PuntDistanceMax', 'PuntReturnedYardsMax'])
print('Official statistics from https://www.pro-football-reference.com:')
reference_data

punt_types = punts.groupby('PuntType')[['PlayKey']].count().reset_index()
punt_types.columns = ['PuntType', 'Cnt']
punt_types = punt_types.sort_values(by='Cnt', ascending=False)
punt_types['RelFreq'] = 100 * punt_types['Cnt'] / punt_types['Cnt'].sum()

data = [
    go.Bar(
        x=punt_types.PuntType.values,
        y=punt_types.RelFreq.values,
        text=punt_types.RelFreq.round(1).values,
        textposition='auto',
        marker=dict(
            color=punt_types['RelFreq'].values,
            colorscale='Viridis',
            showscale=True,
            reversescale=True
        ),
    ),
]
layout = go.Layout(yaxis=dict(title='Share of punts (%)', ticklen=5, gridwidth=2))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PT')

punt_returns = punts[['PuntReturnedYards']].dropna()
punt_returns = punt_returns.sort_values(by='PuntReturnedYards')
yards = np.arange(-10, 100)
ps = [np.mean(punt_returns['PuntReturnedYards'] <= y) for y in yards]

data = [go.Scatter(x=yards, y=ps, mode='lines', line=dict(width=5, color=C[0]))]
layout = go.Layout(
    title='Punt Return Cumulative Density Function',
    xaxis=dict(title='X - Punt return yards', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='P(Return < X)', ticklen=5, gridwidth=2)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PRCDF')

punts.Penalty.sum(), punts.PenaltyType.count()
penalty_types = punts.groupby('PenaltyType')[['PlayKey']].count().reset_index()
penalty_types.columns = ['PenaltyType', 'Cnt']
penalty_types = penalty_types.sort_values(by='Cnt', ascending=False)
penalty_types['RelFreq'] = 100 * penalty_types['Cnt'] / penalty_types['Cnt'].sum()

data = [
    go.Bar(
        x=penalty_types.PenaltyType.values,
        y=penalty_types.Cnt.values,
        marker=dict(
            color=penalty_types['Cnt'].values,
            colorscale='Viridis',
            showscale=True,
            reversescale=True
        ),
    ),
]
layout = go.Layout(
    yaxis=dict(title='Number of penalties (#)', ticklen=5, gridwidth=2, domain=[0.4, 1]))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PT')

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

punt_players = get_punt_players()
punt_players.shape
punt_players.head(2)

players = get_players()
players.shape
players.head()
PlayerRoles = pd.DataFrame([
    ['QB', 'Quarterback'],
    ['RB', 'Running Back'],
    ['FB', 'Fullback'],
    ['WR', 'Wide Receiver'],
    ['TE', 'Tight End'],
    ['OL', 'Offensive Lineman'],
    ['C', 'Center'],
    ['G', 'Guard'],
    ['LG', 'Left Guard'],
    ['RG', 'Right Guard'],
    ['T', 'Tackle'],
    ['LT', 'Left Tackle'],
    ['RT', 'Right Tackle'],
    ['K', 'Kicker'],
    ['KR', 'Kick Returner'],
    ['DL', 'Defensive Lineman'],
    ['DE', 'Defensive End'],
    ['DT', 'Defensive Tackle'],
    ['NT', 'Nose Tackle'],
    ['LB', 'Linebacker'],
    ['ILB', 'Inside Linebacker'],
    ['OLB', 'Outside Linebacker'],
    ['MLB', 'Middle Linebacker'],
    ['DB', 'Defensive Back'],
    ['CB', 'Cornerback'],
    ['FS', 'Free Safety'],
    ['SS', 'Strong Safety'],
    ['S', 'Safety'],
    ['P', 'Punter'],
    ['PR', 'Punt Returner']
], columns=['Abbreviation', 'Position'])
PlayerRoles
print('Source: http://stats.washingtonpost.com/fb/glossary.asp')

YARD = 0.9144
MAXSPEED = 13

def get_ngs(playkey):
    ngs = pd.read_csv(os.path.join(NGS_DIR, f'ngs_{playkey}.csv'), parse_dates=['Time'])
    ngs['t'] = (ngs.Time - ngs.Time.min()) / np.timedelta64(1, 's')
    ngs = ngs.sort_values(by='t')
    return ngs

def calculate_speed_and_acceleration(ngs, smoothing_factor=5):
    speed = ngs.pivot('t', 'GSISID', 'dis') * YARD
    speed = speed.fillna(0)
    speed = speed.rolling(smoothing_factor).mean() * 10
    acc = speed.diff(10)
    return speed, acc

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
    axs[1].set_ylim(-7, 7)
    plt.show()
    fig.savefig(f'speed_profile_{playkey}.png', dpi=300)

playkey = '274_3609'
ngs = get_ngs(playkey)
ngs.shape
ngs.head()
print('We have {} players on the field!'.format(ngs.GSISID.nunique()))
print('The play has movements for {} s'.format(ngs.t.max()))
fig, ax = plt.subplots(figsize=(20, 10))
plt.scatter(ngs.x, ngs.y, c=ngs.t, s=1)
plt.colorbar()
ax.set_xlim(0, 100)
ax.set_ylim(-5, 55)
plt.xticks(range(0, 101, 10))
plt.xlabel('Home Sideline (yards)')
plt.ylabel('Home Endzone (yards)')
plt.grid()
plt.show();
HTML('''<video width="800" height="450" controls>
  <source src="https://s3-eu-west-1.amazonaws.com/nfl-punt-analytics/ryan.mp4" type="video/mp4">
Your browser does not support the video tag.</video>''')
show_injured_player_speed_profile(playkey, 23742, 31785)
ngs_punt = ngs[(ngs.t >= 14) & (ngs.t <= 26)]
fig, ax = plt.subplots(figsize=(20, 10))
plt.scatter(ngs_punt.x - 10, ngs_punt.y, c=ngs_punt.t, s=1)
injured = ngs_punt[ngs_punt.GSISID == 23742]
partner = ngs_punt[ngs_punt.GSISID == 31785]
plt.plot(injured.x - 10, injured.y, c='r', lw=3, alpha=0.2)
plt.plot(partner.x - 10, partner.y, c='k', lw=3, alpha=0.2)
plt.colorbar()
ax.set_xlim(0, 100)
ax.set_ylim(-5, 55)
plt.xticks(range(0, 101, 10))
plt.xlabel('Home Sideline (yards)')
plt.ylabel('Home Endzone (yards)')
plt.grid()
plt.show();

def yard_category(yard):
    try:
        return '{}-{}'.format(int(yard) // 10 * 10, int(yard) // 10 * 10 + 9)
    except Exception as e:
        return None

def get_all_plays():
    data = pd.read_csv(ALL_PLAYS_PATH, parse_dates=['Date'], low_memory=False)
    data['YardLine'] = data['SideofField'] + ' ' + data['yrdln'].astype(str)
    column_mapping = {
        'Date': 'GameDate',
        'qtr': 'Quarter',
        'down': 'Down',
        'time': 'GameClock',
        'yrdline100': 'YardLineTillEndZone100',
        'ydstogo': 'YardsToGo',
        'posteam': 'PossTeam',
        'desc': 'PlayDescription',
        'Penalty.Yards': 'PenaltyYards',
        'AwayTeam': 'VisitTeam',
        'Season': 'SeasonYear',
    }
    additional_columns = [
        'Drive', 'PuntResult', 'PlayType', 'Returner', 'FieldGoalResult', 'TimeSecs',
        'FieldGoalDistance', 'PosTeamScore', 'DefTeamScore', 'HomeTeam', 'YardLine']
    data = data.rename(columns=column_mapping)
    data = data.fillna({'PlayDescription': 'MISSING'})
    data = data[list(column_mapping.values()) + additional_columns]
    data['HomeTeamVisitTeam'] = data['HomeTeam'] + '-' + data['VisitTeam']
    data['YardLine100'] = 100 - data['YardLineTillEndZone100']
    data['YardCat'] = data['YardLine100'].apply(yard_category)
    data['GameId'] = data['GameDate'].apply(lambda d: str(d.date())) + '-' + data[
        'HomeTeamVisitTeam']
    data['PlayId'] = data.groupby('GameId')[['TimeSecs']].rank(ascending=False, method='first')
    data['PlayRankInDrive'] = data.groupby(['GameId', 'Drive'])[['PlayId']].rank(method='first')
    data['Penalty'] = apply_f_for_rows(PlayDescription.is_penalty, data)
    data['PenaltyType'] = apply_f_for_rows(PlayDescription.penalty_type, data)
    return data

def get_all_drives(plays):
    drives = plays[plays.PlayRankInDrive == 1].copy()
    drives['HomeTeamScore'] = (drives['PossTeam'] == drives['HomeTeam']) * \
                              drives['PosTeamScore'] + \
                              (drives['PossTeam'] != drives['HomeTeam']) * \
                              drives['DefTeamScore']
    drives['VisitTeamScore'] = (drives['PossTeam'] != drives['HomeTeam']) * \
                               drives['PosTeamScore'] + \
                               (drives['PossTeam'] == drives['HomeTeam']) * \
                               drives['DefTeamScore']
    drives['HomeTeamDrivePoints'] = -1. * drives['HomeTeamScore'].diff(periods=-1)
    drives['VisitTeamDrivePoints'] = -1. * drives['VisitTeamScore'].diff(periods=-1)
    drives['DriveDuration'] = drives['TimeSecs'].diff(periods=-1)

    drives['PossTeamDrivePoints'] = (2 * (drives['PossTeam'] == drives['HomeTeam']) - 1) * \
                                    (drives['HomeTeamDrivePoints'] -
                                     drives['VisitTeamDrivePoints'])

    drives['NextDrivePossTeam'] = drives['PossTeam'].shift(periods=-1)
    drives['NextDriveGameId'] = drives['GameId'].shift(periods=-1)
    drives['NextDriveYardLine100'] = drives['YardLine100'].shift(periods=-1)
    drives['NextDriveYardCat'] = drives['YardCat'].shift(periods=-1)
    drives = drives[drives['GameId'] == drives['NextDriveGameId']]
    drives = drives[[
        'GameDate', 'Quarter', 'GameClock', 'PossTeam', 'HomeTeam', 'HomeTeamVisitTeam',
        'VisitTeam', 'SeasonYear', 'Drive', 'TimeSecs', 'PosTeamScore', 'DefTeamScore',
        'YardLine100', 'YardCat', 'GameId', 'HomeTeamScore', 'VisitTeamScore',
        'HomeTeamDrivePoints', 'VisitTeamDrivePoints', 'PossTeamDrivePoints', 'DriveDuration',
        'NextDrivePossTeam', 'NextDriveYardLine100', 'NextDriveYardCat']]
    return drives

all_plays = get_all_plays()
all_plays.shape
all_plays.head()
all_plays.to_csv('all_plays.csv', index=False)

drives = get_all_drives(all_plays)
drives.shape
drives.head()
drives.to_csv('drives.csv', index=False)

all_punts = all_plays[all_plays.PlayType == 'Punt'].copy()
all_punts['NextDrive'] = all_punts.Drive + 1
punt_results = all_punts.merge(
    drives, left_on=['GameId', 'NextDrive'],
    right_on=['GameId', 'Drive'], suffixes=['', 'NextDrive'])

punt_results.shape
punt_results[['YardLine100NextDrive', 'PossTeamDrivePoints']].mean()
p = punt_results.groupby('YardCatNextDrive')[['PossTeamDrivePoints']].mean().reset_index()
c = punt_results.groupby('YardCatNextDrive')[['GameId']].count().reset_index()
punt_result_by_yards = pd.merge(p, c, on='YardCatNextDrive')
punt_result_by_yards['Play%'] = 100 * punt_result_by_yards['GameId'] / punt_result_by_yards[
    'GameId'].sum()
punt_result_by_yards

data = [
    go.Bar(
        x=np.arange(0, 100, 10),
        width=[10] * 10,
        y=punt_result_by_yards['Play%'].values,
        text=punt_result_by_yards['Play%'].round(1).values,
        textposition='auto',
        marker=dict(color=C[1]),
        opacity=0.7,
        name='Share of plays'
    ),
    go.Scatter(
        x=np.arange(0, 100, 10),
        y=punt_result_by_yards['PossTeamDrivePoints'].values,
        text=punt_result_by_yards['PossTeamDrivePoints'].round(1).values,
        marker=dict(color=C[0], size=10),
        line=dict(width=3),
        yaxis='y2',
        name='Expected Points'
    ),
]
layout = go.Layout(
    title='Punt results',
    xaxis=dict(title='Punt return team starting position in next drive',
               ticklen=5, gridwidth=2, tickvals=[]),
    yaxis=dict(title='Share of plays (%)', titlefont=dict(color=C[1]),
               tickfont=dict(color=C[1]), ticklen=5, gridwidth=2),
    yaxis2=dict(title='Expected points in next drive', titlefont=dict(color=C[0]),
                tickfont=dict(color=C[0]), overlaying='y', side='right'),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PuntResultBar')
write_image(fig, 'PuntResults.svg')

plays_4th = all_plays[all_plays['Down'] == 4].groupby('PlayType')[
    ['GameDate']].count().reset_index()
plays_4th.columns = ['PlayType', 'Cnt']
plays_4th['RelFreq'] = 100 * plays_4th['Cnt'] / plays_4th['Cnt'].sum()
plays_4th = plays_4th.sort_values(by='Cnt', ascending=False)

data = [
    go.Bar(
        x=plays_4th.PlayType.values,
        y=plays_4th.Cnt.values,
        text=plays_4th.RelFreq.round(1).values,
        textposition='auto',
        marker=dict(
            color=plays_4th['Cnt'].values,
            colorscale='Viridis',
            showscale=True,
            reversescale=True
        ),
    ),
]
layout = go.Layout(title='60% Punt play in 4th',
                   yaxis=dict(title='Share of plays (%)', ticklen=5, gridwidth=2))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='4th')

all_punts = all_plays[all_plays.PlayType == 'Punt'].copy()
all_fgs = all_plays[all_plays.PlayType == 'Field Goal'].copy()

fig, ax = plt.subplots()
sns.distplot(all_punts.YardLine100, bins=20, kde_kws=dict(shade=True), kde=True, color=C[0],
             ax=ax, label='Punt')
sns.distplot(all_fgs.YardLine100, bins=20, kde_kws=dict(shade=True), kde=True, color=C[1],
             ax=ax, label='Field Goal Attempt')
plt.xlim(-5, 105)
plt.xticks(range(0, 101, 10))
plt.legend(loc=0)
plt.ylabel('Probability Density')
plt.title('Punts and Field Goals')
plt.show();

penalties_by_down = all_plays.groupby('Down')[['Penalty']].mean().reset_index()
penalties_by_down['Penalty'] = 100 * penalties_by_down['Penalty']

data = [go.Scatter(
    x=penalties_by_down.Down.values,
    y=penalties_by_down.Penalty.values,
    mode='lines+markers',
    marker=dict(size=10, color=C[0]),
    line=dict(width=3),
    opacity=0.8,
)]
layout = go.Layout(
    title='Penalties are more frequent in last downs',
    xaxis=dict(title='Down', ticklen=5, zeroline=False, gridwidth=2, tickvals=[1, 2, 3, 4]),
    yaxis=dict(title='Chance of penalty (%)', ticklen=5, gridwidth=2),
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='penalty4th')

down4th = all_plays[all_plays['Down'] == 4]
down4th['Punt'] = down4th.PlayDescription.apply(lambda s: 'Punt' if 'punt' in s else 'Rest')
s = down4th.groupby('Punt')['Penalty'].sum()
c = down4th.groupby('Punt')['Penalty'].count()
sc = pd.concat([s, c], axis=1)
sc.columns = ['Penalty', 'Cnt']
sc['Penalty%'] = sc['Penalty'] / sc['Cnt']
sc

down123th = all_plays[all_plays['Down'] < 4]
penalties_in_123 = down123th.groupby(['PenaltyType'])[['Penalty']].count()
penalties_in_123['Rest in 1st - 3rd%'] = 100 * penalties_in_123['Penalty'] / down123th.shape[0]
penalties_in_123 = penalties_in_123.sort_values(by='Rest in 1st - 3rd%', ascending=False)
pt = down4th.groupby(['Punt', 'PenaltyType'])[['GameDate']].count().reset_index()
penalty_types = pt.pivot('PenaltyType', 'Punt', 'GameDate').fillna(0)
penalty_types['Punt%'] = 100 * penalty_types['Punt'] / c['Punt']
penalty_types['Rest in 4th%'] = 100 * penalty_types['Rest'] / c['Rest']
penalty_types = penalty_types.sort_values(by='Punt%', ascending=False)
penalty_types = penalty_types.merge(penalties_in_123[['Rest in 1st - 3rd%']],
                                    how='left', left_index=True, right_index=True)
penalty_types = penalty_types.fillna(0)
penalty_types

serious_penalties = penalty_types[penalty_types.index.isin(SERIOUS_PENALTIES)]
serious_penalties[['Punt%', 'Rest in 4th%', 'Rest in 1st - 3rd%']]
serious_penalties.sum()

data = [
    go.Bar(x=serious_penalties.index.values,
           y=serious_penalties['Punt%'].values,
           marker=dict(color=C[0]),
           name='Punt'),
    go.Bar(x=serious_penalties.index.values,
           y=serious_penalties['Rest in 1st - 3rd%'].values,
           marker=dict(color=C[1]),
           name='1st - 3rd Down'),
    go.Bar(x=serious_penalties.index.values,
           y=serious_penalties['Rest in 4th%'].values,
           marker=dict(color=C[2]),
           name='Rest in 4th Down')
]
layout = go.Layout(
    title='Serious penalties are more often during punts',
    xaxis=dict(ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Chance of Penalty (%)', ticklen=5, gridwidth=2, domain=[0.2, 1]),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='punts-serious')
write_image(fig, 'IBAW.svg')

all_punts.shape
all_punts['Cnt'] = 1
all_punts['SeriousPenalty'] = all_punts.PenaltyType.isin(SERIOUS_PENALTIES)
all_punts['PuntType'] = apply_f_for_rows(PlayDescription.punt_type, all_punts)
all_punts['Injury'] = apply_f_for_rows(PlayDescription.is_injury, all_punts)
all_punts.head()

c = all_punts.groupby('PuntType').count()[['Cnt']].reset_index()
pt = all_punts.groupby('PuntType')[['SeriousPenalty', 'Injury']].mean().reset_index()
punt_penalty_injury = pt.merge(c, how='left', on=['PuntType']).fillna(0)
punt_penalty_injury['SeriousPenalty%'] = 100 * punt_penalty_injury['SeriousPenalty']
punt_penalty_injury['Injury%'] = 100 * punt_penalty_injury['Injury']
punt_penalty_injury = punt_penalty_injury.sort_values(by=['SeriousPenalty%'], ascending=False)
punt_penalty_injury
punt_penalty_injury = punt_penalty_injury[punt_penalty_injury.PuntType != 'OTHER']
all_punts[['Cnt', 'SeriousPenalty', 'Injury']].sum()

punt_penalty_injury = punt_penalty_injury.sort_values(by=['Injury%'], ascending=False)
data = [
    go.Scatter(
        y=punt_penalty_injury['Injury%'].values,
        x=punt_penalty_injury.PuntType.values,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(punt_penalty_injury['Cnt'].values),
                    color=punt_penalty_injury['Injury%'].values,
                    colorscale='Viridis',
                    reversescale=True,
                    showscale=True
                    ),
        text=punt_penalty_injury['Cnt'].values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Injuries per punt play',
    hovermode='closest',
    xaxis=dict(title='Punt Type', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Chance of injury per punt (%)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='InjuryScatter')
write_image(fig, 'InjuryScatter.svg')

punt_penalty_injury = punt_penalty_injury.sort_values(by=['SeriousPenalty%'], ascending=False)
data = [
    go.Scatter(
        y=punt_penalty_injury['SeriousPenalty%'].values,
        x=punt_penalty_injury.PuntType.values,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(punt_penalty_injury['Cnt'].values),
                    color=punt_penalty_injury['SeriousPenalty%'].values,
                    colorscale='Viridis',
                    reversescale=True,
                    showscale=True
                    ),
        text=punt_penalty_injury['Cnt'].values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Serious penalties per punt play',
    hovermode='closest',
    xaxis=dict(title='Punt Type', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Chance of serious penalty per punt (%)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='SeriousPenaltyScatter')
write_image(fig, 'SeriousPenaltyScatter.svg')

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

videos = get_video_review()
video_info = videos.merge(punts, on='PlayKey', how='left')
video_info = video_info.merge(players, on='GSISID', how='left')
video_info = video_info.merge(players, left_on='PrimaryPartnerGSISID', right_on='GSISID',
                              how='left', suffixes=['', 'PrimaryPartner'])
video_info = video_info.merge(punt_players[['GSISID', 'PlayKey', 'ShortRole']],
                              on=['GSISID', 'PlayKey'], how='left')
video_info = video_info.merge(
    punt_players[['GSISID', 'PlayKey', 'ShortRole']],
    left_on=['PrimaryPartnerGSISID', 'PlayKey'],
    right_on=['GSISID', 'PlayKey'],
    how='left',
    suffixes=['', 'PrimaryPartner'])
video_info.to_csv('video_info.csv', index=False)

c = video_info.groupby('PuntType').count()[['PlayID']].reset_index()
pt = punts.groupby('PuntType')[['PlayKey']].count().reset_index()
concussions = pt.merge(c, how='left', on=['PuntType']).fillna(0)
concussions.columns = ['PuntType', '#Punts', '#Concussion']
concussions['Concussion%'] = 100 * concussions['#Concussion'] / concussions['#Punts']
concussions = concussions.sort_values(by=['Concussion%', '#Punts'], ascending=False)
concussions
concussions[['#Punts', '#Concussion']].sum()
concussions = concussions[~concussions.PuntType.isin(['OTHER', 'NOPLAY'])]

data = [
    go.Scatter(
        y=concussions['Concussion%'].values,
        x=concussions.PuntType.values,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(concussions['#Punts'].values),
                    color=concussions['Concussion%'].values,
                    colorscale='Viridis',
                    reversescale=True,
                    showscale=True
                    ),
        text=concussions['#Punts'].values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Known concussions per punt play',
    hovermode='closest',
    xaxis=dict(title='Punt Type', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Chance of concussions per punt (%)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PuntTypeConcussions')
write_image(fig, 'ConcussionScatter.svg')

categories = [
    'PlayerActivityDerived', 'TurnoverRelated', 'PrimaryImpactType',
    'PrimaryPartnerActivityDerived', 'FriendlyFire', 'PuntType', 'Penalty', 'PenaltyType',
    'Injury', 'ShortRole', 'ShortRolePrimaryPartner', 'PuntReturn', 'PuntCoverage',
    'HomeTeamVisitTeam']
video_info['PuntReturn'] = video_info['ShortRole'].apply(lambda s: s in PUNT_RETURN_ROLES)
video_info['PuntCoverage'] = video_info['ShortRole'].apply(lambda s: s in PUNT_COVERAGE_ROLES)
video_info['cnt'] = 1
for c in categories:
    g = video_info[[c, 'cnt']].groupby(c)[['cnt']].count()
    g['cat'] = c
    g = g.sort_values(by='cnt', ascending=False)
    g

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))