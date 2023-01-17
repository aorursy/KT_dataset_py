%matplotlib inline
import os
import sys
import re
import pandas as pd
import numpy as np
import glob
import os
import logging
import sys
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  
warnings.filterwarnings('ignore')

ngs_files = ['../input/NGS-2016-pre.csv',
             '../input/NGS-2016-reg-wk1-6.csv',
             '../input/NGS-2016-reg-wk7-12.csv',
             '../input/NGS-2016-reg-wk13-17.csv',
             '../input/NGS-2016-post.csv',
             '../input/NGS-2017-pre.csv',
             '../input/NGS-2017-reg-wk1-6.csv',
             '../input/NGS-2017-reg-wk7-12.csv',
             '../input/NGS-2017-reg-wk13-17.csv',
             '../input/NGS-2017-post.csv']
PUNT_TEAM = set(['GL', 'PLW', 'PLT', 'PLG', 'PLS', 'PRG', 'PRT', 'PRW', 'PC',
                 'PPR', 'P', 'GR'])
RECV_TEAM = set(['VR', 'PDR', 'PDL', 'PLR', 'PLM', 'PLL', 'VL', 'PFB', 'PR'])


plays_df = pd.read_csv('../input/play_information.csv')

def get_return_yards(s):
    m = re.search('for ([0-9]+) yards', s)
    if m:
        return int(m.group(1))
    elif re.search('for no gain', s):
        return 0
    else:
        return np.nan

plays_df['Return'] = plays_df['PlayDescription'].map(
        lambda x: get_return_yards(x))

video_review = pd.read_csv('../input/video_review.csv')
video_review = video_review.rename(columns={'GSISID': 'InjuredGSISID'})

plays_df= plays_df.merge(video_review, how='left',
                         on=['Season_Year', 'GameKey', 'PlayID'])

plays_df['InjuryOnPlay'] = 0
plays_df.loc[plays_df['InjuredGSISID'].notnull(), 'InjuryOnPlay'] = 1

plays_df = plays_df[['Season_Year', 'GameKey', 'PlayID', 'Return', 'InjuryOnPlay']]

ngs_df = []
for filename in ngs_files:
    df = pd.read_csv(filename, parse_dates=['Time'])
    df = df.loc[df['Event'].isin(['fair_catch', 'punt_received'])]
    df = pd.concat([df, pd.get_dummies(df['Event'])], axis=1)
    df = df.groupby(['Season_Year', 'GameKey', 'PlayID'])[['fair_catch', 'punt_received']].max()
    ngs_df.append(df.reset_index())
ngs_df = pd.concat(ngs_df)

plays_df = plays_df.merge(ngs_df, on=['Season_Year', 'GameKey', 'PlayID'])
injury_per_1000_fair_catch = 1000 * plays_df.loc[plays_df['fair_catch']==1,
                                          'InjuryOnPlay'].mean()
injury_per_1000_punt_received = 1000 * plays_df.loc[plays_df['punt_received']==1,
                                           'InjuryOnPlay'].mean()
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar([0, 1], [injury_per_1000_fair_catch, injury_per_1000_punt_received])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fair Catch', 'Punt Received'])
plt.text(0, injury_per_1000_fair_catch+0.2, '{:.1f}'.format(injury_per_1000_fair_catch))
plt.text(1, injury_per_1000_punt_received+0.2, '{:.1f}'.format(injury_per_1000_punt_received))
plt.title("Concussion Rate")
plt.ylabel("Injuries per 1000 Events")
sns.despine(top=True, right=True)
plt.show()
x_groups = ['0-3 yds', '3-5 yds', '5-7 yds', '7-9 yds',
            '9-12 yds', '12-15 yds', '15-20 yds', '20+ yds']
rec = plays_df.loc[(plays_df['punt_received']==1) 
                   &(plays_df['Return'].notnull())]

y_groups = [sum(rec['Return']<=3) / len(rec),
            sum((rec['Return']>3) & (rec['Return']<=5)) / len(rec),
            sum((rec['Return']>5) & (rec['Return']<=7)) / len(rec),
            sum((rec['Return']>7) & (rec['Return']<=9)) / len(rec),
            sum((rec['Return']>9) & (rec['Return']<=12)) / len(rec),
            sum((rec['Return']>12) & (rec['Return']<=15)) / len(rec),
            sum((rec['Return']>15) & (rec['Return']<=20))/ len(rec),
            sum(rec['Return']>20) / len(rec)]

y_bottoms = [0,
             sum(rec['Return']<=3) / len(rec),
             sum(rec['Return']<=5) / len(rec),
             sum(rec['Return']<=7) / len(rec),
             sum(rec['Return']<=9) / len(rec),
             sum(rec['Return']<=12) / len(rec),
             sum(rec['Return']<=15) / len(rec),
             sum(rec['Return']<=20) / len(rec)]

fig = plt.figure(figsize=(8.5,4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar(range(len(x_groups)), y_groups, bottom=y_bottoms)
ax.set_xticks(range(len(x_groups)))
ax.set_xticklabels(x_groups)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
for i in range(len(x_groups)):
    plt.text(i-0.2, y_bottoms[i]+y_groups[i]+0.02, '{:.0f}%'.format(100*y_groups[i]))
sns.despine(top=True, right=True)
plt.title("Distribution of Punt Returns by Length")
plt.show()
play_player_role_data = pd.read_csv('../input/play_player_role_data.csv')
gfp = video_review.merge(play_player_role_data, on=['GameKey', 'PlayID', 'Season_Year'])

df_29 = pd.read_csv('../input/NGS-2016-pre.csv', parse_dates=['Time'])
df_29 = df_29.loc[(df_29['GameKey']==29) & (df_29['PlayID']==538)]
df_29 = df_29.merge(gfp, on=['GameKey', 'PlayID', 'Season_Year', 'GSISID'])
df_29 = df_29.sort_values(['GameKey', 'PlayID', 'Season_Year', 'GSISID', 'Time'])

fig = plt.figure(figsize = (10, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
line_set = df_29.loc[df_29['Event'].isin(['ball_snap'])]
line_set_time = line_set['Time'].min()
line_of_scrimmage = line_set.loc[line_set['Role'].isin(['PLS', 'PLG', 'PRG']), 'x'].median()

recv_df = df_29.loc[df_29['Event']=='punt_received']
recv_time = recv_df['Time'].min()
event_df = df_29.loc[df_29['Time'] <= recv_time]
event_df = event_df.loc[event_df['Time'] >= recv_time + pd.Timedelta('-2s')]

injured = df_29['InjuredGSISID'].values[0]
partner = float(df_29['Primary_Partner_GSISID'].values[0])
            
players = event_df['GSISID'].unique()
for player in players:
    player_df = event_df.loc[event_df['GSISID'] == player]
    role = str(player_df['Role'].values[0])
    if re.sub('[io0-9]', '', str(role)) in PUNT_TEAM:
        color = '#fdc086'
        marker = 'x'
        linewidth = 6
    else:
        color = '#beaed4'
        marker = 'o'
        linewidth = 6
    if player == injured:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#f0027f'
    elif player == partner:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#ffff99'
    else:
        linestyle = '--'
    alphas = np.ones(len(player_df))
    alphas = alphas.cumsum() / alphas.sum()
    px = player_df['x'].values
    py = player_df['y'].values
    for k in range(len(px)):
        plt.plot(px[k:], py[k:], color=color,
                 linewidth=linewidth*(k+1+4)/(4+len(px)),
                 linestyle=linestyle,
                 alpha=(k+1)/len(px))
        player_df = player_df.reset_index(drop=True)
        x = player_df['x'].iloc[-1]
        y = player_df['y'].iloc[-1]

        marker = (3, 0, 90 + player_df['o'].iloc[-1])
        plt.scatter(player_df['x'].iloc[-1],
                    player_df['y'].iloc[-1],
                    marker=marker,
                    s=linewidth*60,
                    color=color)
        if (role == 'PR'):
            circ = plt.Circle((player_df['x'].iloc[-1],
                                player_df['y'].iloc[-1]),
                                5, color=color,
                                fill=False)
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
plt.axvline(x=0, color='w', linewidth=2)
plt.axvline(x=10, color='w', linewidth=2)
plt.axvline(x=15, color='w', linewidth=2)
plt.axvline(x=20, color='w', linewidth=2)
plt.axvline(x=25, color='w', linewidth=2)
plt.axvline(x=30, color='w', linewidth=2)
plt.axvline(x=35, color='w', linewidth=2)
plt.axvline(x=40, color='w', linewidth=2)
plt.axvline(x=45, color='w', linewidth=2)
plt.axvline(x=50, color='w', linewidth=2)
plt.axvline(x=55, color='w', linewidth=2)
plt.axvline(x=60, color='w', linewidth=2)
plt.axvline(x=65, color='w', linewidth=2)
plt.axvline(x=70, color='w', linewidth=2)
plt.axvline(x=75, color='w', linewidth=2)
plt.axvline(x=80, color='w', linewidth=2)
plt.axvline(x=85, color='w', linewidth=2)
plt.axvline(x=90, color='w', linewidth=2)
plt.axvline(x=95, color='w', linewidth=2)
plt.axvline(x=100, color='w', linewidth=2)
plt.axvline(x=105, color='w', linewidth=2)
plt.axvline(x=110, color='w', linewidth=2)
plt.axvline(x=120, color='w', linewidth=2)
plt.axvline(x=line_of_scrimmage, color='y', linewidth=3)
plt.axhline(y=0, color='w', linewidth=2)
plt.axhline(y=53.3, color='w', linewidth=2)
plt.text(x=18, y=2, s= '1', color='w')
plt.text(x=21, y=2, s= '0', color='w')
plt.text(x=28, y=2, s= '2', color='w')
plt.text(x=31, y=2, s= '0', color='w')
plt.text(x=38, y=2, s= '3', color='w')
plt.text(x=41, y=2, s= '0', color='w')
plt.text(x=48, y=2, s= '4', color='w')
plt.text(x=51, y=2, s= '0', color='w')
plt.text(x=58, y=2, s= '5', color='w')
plt.text(x=61, y=2, s= '0', color='w')
plt.text(x=68, y=2, s= '4', color='w')
plt.text(x=71, y=2, s= '0', color='w')
plt.text(x=78, y=2, s= '3', color='w')
plt.text(x=81, y=2, s= '0', color='w')
plt.text(x=88, y=2, s= '2', color='w')
plt.text(x=91, y=2, s= '0', color='w')
plt.text(x=98, y=2, s= '1', color='w')
plt.text(x=101, y=2, s= '0', color='w')

ax.set_xticks([0, 120])
ax.set_yticks([0, 53.3])
ax.set_xticklabels(['', ''])
ax.set_yticklabels(['', ''])
ax.tick_params(axis=u'both', which=u'both', length=0)
ax.add_artist(circ)
ax.set_facecolor("#2ca25f")
plt.title("GR is injured while tackling PR (helmet to body).\nGameKey 29, Play ID 538. NYJ Punting to WAS.")
plt.show() 
df_296 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv', parse_dates=['Time'])
df_296 = df_296.loc[(df_296['GameKey']==296) & (df_296['PlayID']==2667)]
df_296 = df_296.merge(gfp, on=['GameKey', 'PlayID', 'Season_Year', 'GSISID'])
df_296 = df_296.sort_values(['GameKey', 'PlayID', 'Season_Year', 'GSISID', 'Time'])
fig = plt.figure(figsize = (10, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
line_set = df_296.loc[df_296['Event'].isin(['ball_snap'])]
line_set_time = line_set['Time'].min()
line_of_scrimmage = line_set.loc[line_set['Role'].isin(['PLS', 'PLG', 'PRG']), 'x'].median()

recv_df = df_296.loc[df_296['Event']=='punt_received']
recv_time = recv_df['Time'].min()
event_df = df_296.loc[df_296['Time'] <= recv_time]
event_df = event_df.loc[event_df['Time'] >= recv_time + pd.Timedelta('-2s')]

injured = df_296['InjuredGSISID'].values[0]
partner = float(df_296['Primary_Partner_GSISID'].values[0])
            
players = event_df['GSISID'].unique()
for player in players:
    player_df = event_df.loc[event_df['GSISID'] == player]
    role = str(player_df['Role'].values[0])
    if re.sub('[io0-9]', '', str(role)) in PUNT_TEAM:
        color = '#fdc086'
        marker = 'x'
        linewidth = 6
    else:
        color = '#beaed4'
        marker = 'o'
        linewidth = 6
    if player == injured:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#f0027f'
    elif player == partner:
        marker = '*'
        linewidth = 10
        linestyle = '-'
        color = '#ffff99'
    else:
        linestyle = '--'
    alphas = np.ones(len(player_df))
    alphas = alphas.cumsum() / alphas.sum()
    px = player_df['x'].values
    py = player_df['y'].values
    for k in range(len(px)):
        plt.plot(px[k:], py[k:], color=color,
                 linewidth=linewidth*(k+1+4)/(4+len(px)),
                 linestyle=linestyle,
                 alpha=(k+1)/len(px))
        player_df = player_df.reset_index(drop=True)
        x = player_df['x'].iloc[-1]
        y = player_df['y'].iloc[-1]

        marker = (3, 0, 90 + player_df['o'].iloc[-1])
        plt.scatter(player_df['x'].iloc[-1],
                    player_df['y'].iloc[-1],
                    marker=marker,
                    s=linewidth*60,
                    color=color)
        if (role == 'PR'):
            circ = plt.Circle((player_df['x'].iloc[-1],
                                player_df['y'].iloc[-1]),
                                5, color=color,
                                fill=False)
ax.set_xlim(0, 120)
ax.set_ylim(0, 53.3)
plt.axvline(x=0, color='w', linewidth=2)
plt.axvline(x=10, color='w', linewidth=2)
plt.axvline(x=15, color='w', linewidth=2)
plt.axvline(x=20, color='w', linewidth=2)
plt.axvline(x=25, color='w', linewidth=2)
plt.axvline(x=30, color='w', linewidth=2)
plt.axvline(x=35, color='w', linewidth=2)
plt.axvline(x=40, color='w', linewidth=2)
plt.axvline(x=45, color='w', linewidth=2)
plt.axvline(x=50, color='w', linewidth=2)
plt.axvline(x=55, color='w', linewidth=2)
plt.axvline(x=60, color='w', linewidth=2)
plt.axvline(x=65, color='w', linewidth=2)
plt.axvline(x=70, color='w', linewidth=2)
plt.axvline(x=75, color='w', linewidth=2)
plt.axvline(x=80, color='w', linewidth=2)
plt.axvline(x=85, color='w', linewidth=2)
plt.axvline(x=90, color='w', linewidth=2)
plt.axvline(x=95, color='w', linewidth=2)
plt.axvline(x=100, color='w', linewidth=2)
plt.axvline(x=105, color='w', linewidth=2)
plt.axvline(x=110, color='w', linewidth=2)
plt.axvline(x=120, color='w', linewidth=2)
plt.axvline(x=line_of_scrimmage, color='y', linewidth=3)
plt.axhline(y=0, color='w', linewidth=2)
plt.axhline(y=53.3, color='w', linewidth=2)
plt.text(x=18, y=2, s= '1', color='w')
plt.text(x=21, y=2, s= '0', color='w')
plt.text(x=28, y=2, s= '2', color='w')
plt.text(x=31, y=2, s= '0', color='w')
plt.text(x=38, y=2, s= '3', color='w')
plt.text(x=41, y=2, s= '0', color='w')
plt.text(x=48, y=2, s= '4', color='w')
plt.text(x=51, y=2, s= '0', color='w')
plt.text(x=58, y=2, s= '5', color='w')
plt.text(x=61, y=2, s= '0', color='w')
plt.text(x=68, y=2, s= '4', color='w')
plt.text(x=71, y=2, s= '0', color='w')
plt.text(x=78, y=2, s= '3', color='w')
plt.text(x=81, y=2, s= '0', color='w')
plt.text(x=88, y=2, s= '2', color='w')
plt.text(x=91, y=2, s= '0', color='w')
plt.text(x=98, y=2, s= '1', color='w')
plt.text(x=101, y=2, s= '0', color='w')

ax.set_xticks([0, 120])
ax.set_yticks([0, 53.3])
ax.set_xticklabels(['', ''])
ax.set_yticklabels(['', ''])
ax.tick_params(axis=u'both', which=u'both', length=0)
ax.add_artist(circ)
ax.set_facecolor("#2ca25f")
plt.title("GL and GR collide, injuring GL (helmet to helmet friendly fire).\nGameKey 296, Play ID 2667. TEN punting to JAX.")
plt.show()
ppr = pd.read_csv('../input/play_player_role_data.csv')
ppr['Role'] = ppr['Role'].map(lambda x: re.sub('[oi0-9]', '', x))
roles = ppr['Role'].unique()

ppr = pd.concat([ppr, pd.get_dummies(ppr['Role'])], axis=1)
ppr = ppr.groupby(['Season_Year', 'GameKey', 'PlayID'])[roles].sum()
ppr = ppr.reset_index()

vi = pd.read_csv('../input/video_review.csv')
vi = vi[['Season_Year', 'GameKey', 'PlayID', 'GSISID']]

ppr = ppr.merge(vi, on=['Season_Year', 'GameKey', 'PlayID'], how='left')
ppr['Injury'] = 0
ppr['const'] = 1

ppr.loc[ppr['GSISID'].notnull(), 'Injury'] = 1

play_information = pd.read_csv('../input/play_information.csv')
def extract_recv_yards(s):
    m = re.search('for ([0-9]+) yards', s)
    if m:
        return int(m.group(1))
    elif re.search('for no gain', s):
        return 0
    else:
        return np.nan
play_information['recv_length'] = play_information['PlayDescription'].map(
        lambda x: extract_recv_yards(x))
play_information = play_information[['Season_Year', 'GameKey', 'PlayID', 'YardLine', 'Poss_Team', 'recv_length']]
play_information['yards_to_go'] = play_information['YardLine'].map(lambda x: int(x[-2:]))
play_information['back_half'] = play_information.apply(lambda x:
    x['YardLine'].startswith(x['Poss_Team']), axis = 1)
play_information.loc[play_information['back_half']==1, 'yards_to_go'] = 100 - (
    play_information.loc[play_information['back_half']==1, 'yards_to_go'])

play_information = play_information[['Season_Year', 'GameKey', 'PlayID', 'yards_to_go', 'recv_length']]
ppr = ppr.merge(play_information, on=['Season_Year', 'GameKey', 'PlayID'], how='inner')

col = 'VR_VL'
ppr[col] = ppr.loc[:, ['VR', 'VL']].sum(axis=1)
ptiles = np.percentile(ppr[col], [0, 25, 50, 75, 100])

ppr['yards'] = ''
ppr['coverage'] = ''
ppr.loc[ppr[col]==2, 'coverage'] = 'single'
ppr.loc[ppr[col]==3, 'coverage'] = 'hybrid'
ppr.loc[ppr[col]==4, 'coverage'] = 'double'

### For graphing, keep track of counts of plays by single, hybrid, or double coverage
### Sort by yards-to-go.
x_single = [0.00, 1.00, 2.00]
x_hybrid = [0.25, 1.25, 2.25]
x_double = [0.50, 1.50, 2.50]
y_single = []
y_hybrid = []
y_double = []
injury_rate = {'Yards to Go': [], '30-50': [], '50-70': [], '70-100': []}
for i in range(2, 5):
    if i == 2:
        mode = 'Single'
    elif i == 3:
        mode = 'Hybrid'
    elif i == 4:
        mode = 'Double'
    injury_rate['Yards to Go'].append(mode)
    for r in ([30, 50], [50, 70], [70, 100]):
        ii = (ppr['yards_to_go']>=r[0])&(ppr['yards_to_go']<r[1])
        if (r[0]==30) & (r[1]==50):
            ppr.loc[ii, 'yards'] = '30 to 50'
        elif (r[0]==50) & (r[1]==70):
            ppr.loc[ii, 'yards'] = '50 to 70'
        elif (r[0]==70) & (r[1]==100):
            ppr.loc[ii, 'yards'] = '70 to 100'

        pprt = ppr.loc[ii]
        if len(pprt) == 0:
            pass
        iii = (pprt[col] == i)
        if sum(iii) == 0:
            continue
        # Keep track of coverage choice by yards to go
        if mode == 'Single':
            y_single.append(sum(iii) / sum(ii)) # Ratio of times single coverage is elected
        elif mode == 'Hybrid':
            y_hybrid.append(sum(iii) / sum(ii)) # Ratio of times hybrid coverage is elected
        elif mode == 'Double':
            y_double.append(sum(iii) / sum(ii)) # Ratio of times double coverage is elected
        if r[0] == 30:
            injury_rate['30-50'].append(1000 * pprt.loc[iii, 'Injury'].mean())
        elif r[0] == 50:
            injury_rate['50-70'].append(1000 * pprt.loc[iii, 'Injury'].mean())
        elif r[0] == 70:
            injury_rate['70-100'].append(1000 * pprt.loc[iii, 'Injury'].mean())
            
injury_rate['Yards to Go'].append("All")
injury_rate['30-50'].append(1000 * ppr.loc[
    (ppr['yards_to_go']>=30) & (ppr['yards_to_go']<50), 'Injury'].mean())
injury_rate['50-70'].append(1000 * ppr.loc[
    (ppr['yards_to_go']>=50) & (ppr['yards_to_go']<70), 'Injury'].mean())
injury_rate['70-100'].append(1000 * ppr.loc[
    (ppr['yards_to_go']>=70) & (ppr['yards_to_go']<100), 'Injury'].mean())
injury_rate = pd.DataFrame(injury_rate)
injury_rate = injury_rate[['Yards to Go', '30-50', '50-70', '70-100']]

fig = plt.figure(figsize = (6.5, 4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar(x_single, y_single, color='#7fc97f', width=.25, label='Single Coverage')
plt.bar(x_hybrid, y_hybrid, color='#beaed4', width=.25, label='Hybrid Coverage')
plt.bar(x_double, y_double, color='#fdc086', width=.25, label='Double Coverage')
for i in range(3):
    plt.text(x_single[i]-0.06, y_single[i]+0.02, '{:.0f}%'.format(y_single[i] * 100))
    plt.text(x_hybrid[i]-0.06, y_hybrid[i]+0.02, '{:.0f}%'.format(y_hybrid[i] * 100))
    plt.text(x_double[i]-0.06, y_double[i]+0.02, '{:.0f}%'.format(y_double[i] * 100))
    
plt.legend()
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.00])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xticks([0.25, 1.25, 2.25])
ax.set_xticklabels(['30-50 Yards', '50-70 Yards', '70-100 Yards'])
ax.set_xlabel('Yards from Line of Scrimmage to End Zone')
ax.set_ylabel('Coverage Choice (%)')
sns.despine(top=True, right=True)

plt.title("Coverage Choice vs. Yards between Line of Scrimmage and End Zone\n")
plt.show()
injury_rate.style.set_precision(2).hide_index()
fig = plt.figure(figsize = (8.5, 5.5))
ax = plt.subplot2grid((1, 1), (0, 0))
pal = {'single': '#7fc97f', 'hybrid': '#beaed4', 'double': '#fdc086'}
ax = sns.violinplot(x="yards", y="recv_length", hue="coverage",
                    data=ppr, palette=pal,
                    order=['30 to 50', '50 to 70', '70 to 100'],
                    hue_order=['single', 'hybrid', 'double'],
                    cut=0)
ax.legend().remove()
sns.despine(top=True, right=True)
plt.legend()
plt.show()
# Collate data
game_data = pd.read_csv('../input/game_data.csv')
play_information = pd.read_csv('../input/play_information.csv')
player_punt_data = pd.read_csv('../input/player_punt_data.csv')
player_punt_data = player_punt_data.groupby('GSISID').head(1).reset_index()
play_player_role_data = pd.read_csv('../input/play_player_role_data.csv')
video_review = pd.read_csv('../input/video_review.csv')

combined = game_data.merge(play_information.drop(['Game_Date'], axis=1),
                        on=['GameKey', 'Season_Year', 'Season_Type', 'Week'])
combined = combined.merge(play_player_role_data,
                        on=['GameKey', 'Season_Year', 'PlayID'])
combined = combined.merge(player_punt_data, on=['GSISID'])

combined = combined.merge(video_review, how='left',
                          on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'])
combined['injury'] = 0
combined.loc[combined['Player_Activity_Derived'].notnull(), 'injury'] = 1

ngs_files = ['../input/NGS-2016-pre.csv',
             '../input/NGS-2016-reg-wk1-6.csv',
             '../input/NGS-2016-reg-wk7-12.csv',
             '../input/NGS-2016-reg-wk13-17.csv',
             '../input/NGS-2016-post.csv',
             '../input/NGS-2017-pre.csv',
             '../input/NGS-2017-reg-wk1-6.csv',
             '../input/NGS-2017-reg-wk7-12.csv',
             '../input/NGS-2017-reg-wk13-17.csv',
             '../input/NGS-2017-post.csv']

max_decel_df = []
for filename in ngs_files:
    logging.info("Loading file " + filename)
    group_keys = ['Season_Year', 'GameKey', 'PlayID', 'GSISID']
    df = pd.read_csv(filename, parse_dates=['Time'])
    logging.info("Read file " + filename)

    df = df.sort_values(group_keys + ['Time'])
    df['dx'] = df.groupby(group_keys)['x'].diff(1)
    df['dy'] = df.groupby(group_keys)['y'].diff(1)
    df['dis'] = (df['dx']**2 + df['dy']**2)**0.5
    df['dt'] = df.groupby(group_keys)['Time'].diff(1).dt.total_seconds()
    df['velocity'] = 0
    ii = (df['dis'].notnull() & df['dt'].notnull() & (df['dt']>0))
    df.loc[ii, 'velocity'] = df.loc[ii, 'dis'] / df.loc[ii, 'dt']
    df['velocity'] *= 0.9144 # Convert yards to meters
    df['deceleration'] = -1 * df.groupby(group_keys)['velocity'].diff(1)
    df['velocity'] = df.groupby(group_keys)['velocity'].shift(1)

    # Only look at the one second window around each tackle
    df['Event'] = df.groupby(group_keys)['Event'].ffill(limit=5)
    df['Event'] = df.groupby(group_keys)['Event'].bfill(limit=5)

    t_df = df.loc[df['Event']=='tackle']

    t_max_decel = t_df.loc[t_df.groupby(['Season_Year', 'GameKey', 'PlayID', 'GSISID'])['deceleration'].idxmax()]
    t_max_decel = t_max_decel[['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'deceleration']].rename(columns={'deceleration': 'deceleration_at_tackle'})
    
    t_max_velocity = t_df.loc[t_df.groupby(['Season_Year', 'GameKey', 'PlayID', 'GSISID'])['velocity'].idxmax()]
    t_max_velocity = t_max_velocity[['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'velocity']].rename(columns={'velocity': 'velocity_at_tackle'})

    max_decel = t_max_velocity.merge(t_max_decel, on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'],
                                      how='outer')
    max_decel_df.append(max_decel)

max_decel_df = pd.concat(max_decel_df)
combined = combined.merge(max_decel_df, on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'], how='left')


combined['tackle_injury'] = combined['Player_Activity_Derived'].isin(['Tackled', 'Tackling'])
### Original work by Halla Yang
fig = plt.figure(figsize = (6.5, 5.0))
ax = plt.subplot2grid((1, 1), (0, 0))
inj = combined.loc[(combined['injury']==1)&(combined['velocity_at_tackle'].notnull())
                  &(combined['deceleration_at_tackle'].notnull())]
ax = sns.kdeplot(inj.velocity_at_tackle, inj.deceleration_at_tackle,
                 cmap="Reds")
notinj = combined.loc[(combined['injury']==0)&(combined['velocity_at_tackle'].notnull())
                  &(combined['deceleration_at_tackle'].notnull())]
ax = sns.kdeplot(notinj.velocity_at_tackle, notinj.deceleration_at_tackle,
                 cmap="Blues")
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
plt.xlabel("Velocity at Tackle")
plt.ylabel("Deceleration at Tackle")
sns.despine(top=True, right=True)
plt.title("Velocity/Deceleration at Time of Tackle for Injuries shown in Red, Non-Injuries in Blue")
plt.subplots_adjust(top=0.9)
plt.show()
