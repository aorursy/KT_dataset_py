# Import packages

import numpy as np

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import matplotlib.gridspec as gridspec

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Data info

df_columns = pd.read_csv('../input/_Columns.csv',sep=',')

df_raw = pd.read_csv('../input/_LeagueofLegends.csv',sep=',')

df_raw.info()
# What the data looks like

df_raw.head(1)
# Copying df_raw to keep it unmodified and adding some columns to df

df = df_raw.copy(deep=True)

df['win_team'] = np.where(df['bResult']==1, 'blue', 'red')

df[['win_team', 'bResult', 'rResult']].head()
# Setstyle options

sns.set_style('whitegrid')

sns.palplot(sns.color_palette('Blues', 20))

colors = sns.color_palette('Blues', 20)



# Create Figure

fig, ax = plt.subplots(2,4, figsize=(16,14))

fig.suptitle('Game Length Distribution', x=0.065, y=1.03, fontsize=24, fontweight='bold', 

             horizontalalignment='left')

fig.subplots_adjust(top=0.9)



percentiles = np.array([25, 50, 75])

ptiles_gl = np.percentile(df['gamelength'], percentiles)



# Create Subplots



# 1 Box and Whisker

p1 = plt.subplot2grid((2,4), (0,0), colspan=1)

sns.boxplot(y=df['gamelength'], color=colors[14])

# Swarm plot adds no value here, ignore below

# sns.swarmplot(y=df['gamelength'], color=colors[2])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel('All Games', fontsize=18)

plt.ylabel('Minutes', fontsize = 18, fontweight = 'bold')



# 2 ECDF Plot

p2 = plt.subplot2grid((2,4), (0,1), colspan=3)

x = np.sort(df['gamelength'])

y = np.arange(1, len(x) + 1) / len(x)

plt.plot(x,y, marker='.', linestyle='none', color=colors[16])

plt.plot(ptiles_gl, percentiles/100, marker='D', color='red', linestyle='none')



# 2 ECDF Formatting (a lot)

yvals = p2.get_yticks()

p2.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

plt.yticks(fontsize=14)

plt.xticks(np.arange(0, 85, 5), fontsize=14)

plt.xlabel('Minutes', fontsize=18, fontweight = 'bold')

plt.ylabel('ECDF', fontsize=18, fontweight='bold')

plt.margins(0.02)



plt.annotate('25% of games were 32 minutes or less', xy=(32, .25), xytext=(37, .23), fontsize=18, 

             arrowprops=dict(facecolor='black'))

plt.annotate('50% of games were 37 minutes or less', xy=(37, .5), xytext=(42, .48), 

             fontsize=18, arrowprops=dict(facecolor='black'))

plt.annotate('75% of games were 42 minutes or less', xy=(42, .75), xytext=(47, .73), fontsize=18, 

             arrowprops=dict(facecolor='black'))



# 3 Histogram Count

p3 = plt.subplot2grid((2,4), (1,0), colspan=2)

plt.hist(x='gamelength', bins=80, data=df, color=colors[14])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel('Minutes', fontweight = 'bold', fontsize = 18)

plt.ylabel('Count of All Games', fontsize=18, fontweight='bold')



# 3 Histogram Percentage - Second Y Axis for Percent (To DO - align Y2 ytick values to Y1 ytick lines)

weights = np.ones_like(df['gamelength']) / len(df['gamelength'])

p3 = plt.twinx()

plt.hist(x='gamelength', bins=80, weights= weights, data=df, color=colors[14])

yvals = p3.get_yticks()

p3.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

p3.grid(b=False)



# 4 Distribution Plot across Years

p4 = plt.subplot2grid((2,4), (1,2), colspan=2)

sns.distplot((df['gamelength'][df['Year']==2014]), hist=False, color='r', label='2014')

sns.distplot((df['gamelength'][df['Year']==2015]), hist=False, color='grey', label='2015')

sns.distplot((df['gamelength'][df['Year']==2016]), hist=False, color='y', label='2016')

sns.distplot((df['gamelength'][df['Year']==2017]), hist=False, color='g', label='2017')

sns.distplot((df['gamelength']), hist=False, color='b', label='All Years')

# Formatting

yvals = p4.get_yticks()

p4.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.ylabel('Percent of All Games\n', fontsize=18, fontweight='bold')

plt.xlabel('Minutes', fontsize = 18, fontweight = 'bold')



# Show everything

plt.tight_layout()

plt.show()
pvt_wins_y = df.pivot_table(index = 'Year', values = ['bResult', 'rResult'], aggfunc = np.sum,  

                                margins = False)

pvt_wins_y['b_net_wins'] = pvt_wins_y['bResult'] - pvt_wins_y['rResult']

pvt_wins_y['b_win_pcnt'] = pvt_wins_y['bResult'] / (pvt_wins_y['bResult'] + pvt_wins_y['rResult'])

pvt_wins_y['b_pcnt_diff'] = pvt_wins_y['b_win_pcnt'] -.5

pvt_wins_y
sns.palplot(sns.color_palette('RdBu', 20))
# Set plot styles and colors

blues = sns.color_palette('Blues', 4)

reds = sns.color_palette('Reds', 4)



# Control y 

y_max = 1.1 * max(max(pvt_wins_y['bResult']), max(pvt_wins_y['rResult']))



fig, axes = plt.subplots(1,3, figsize=(14,6))

fig.suptitle('Wins Over Time', x=0.125, y=1.03, fontsize=24, fontweight='bold', 

             horizontalalignment='left')



# Blue Total Wins Plot

plt.subplot(1,3,1)

sns.barplot(x=pvt_wins_y.index, y='bResult', data = pvt_wins_y, palette=[blues[0], blues[2], blues[3], blues[1]])

plt.title('Blue Total Wins', fontsize=14)

plt.ylim(0, y_max)

plt.ylabel('Count', fontsize = 14, fontweight = 'bold')



# Red Total Wins Plot

plt.subplot(1,3,2)

sns.barplot(x=pvt_wins_y.index, y='rResult', data = pvt_wins_y, palette=[reds[0], reds[2], reds[3], reds[1]])

plt.title('Red Total Wins', fontsize=14)

plt.ylim(0, y_max)

plt.ylabel('')



# Blue Net Wins Plot

plt.subplot(1,3,3)

sns.barplot(x=pvt_wins_y.index, y='b_net_wins', data = pvt_wins_y, palette=[blues[0], blues[2], blues[1], blues[3]])

plt.title('Blue Net Wins', fontsize=14)

plt.ylim(0, y_max)

plt.ylabel('')



plt.show()
pd.unique(df['League'])
dct_leagues = {'North_America':'NA', 'Europe':'EUR', 'LCK':'LCK', 'LMS':'LMS', 'Season_World_Championship':'SWC', 

               'Mid-Season_Invitational':'MSI', 'CBLOL':'CBLOL'}



# Map League Abbreviations

df['LA'] = df['League'].map(dct_leagues)

print(pd.unique(df['LA']))

df[['League', 'LA']].head()
# Pivot by Year and League

pvt_net_wins_yl = df.pivot_table(index = ['Year', 'LA'], values = ['bResult', 'rResult'], aggfunc=np.sum)

pvt_net_wins_yl['b_net_wins'] = pvt_net_wins_yl['bResult'] - pvt_net_wins_yl['rResult']

pvt_net_wins_yl['positive'] = pvt_net_wins_yl['b_net_wins'] > 0



# Color Formatting

blues = sns.color_palette('Blues')

reds = sns.color_palette('Reds')



lst_years = pd.unique(pvt_net_wins_yl.index.get_level_values(0))

lst_x = [1, 2, 3, 4]



y_max = 1.25 * pvt_net_wins_yl['b_net_wins'].max()

y_min = 1.25 * pvt_net_wins_yl['b_net_wins'].min()



fig, ax = plt.subplots(2,2, figsize = (14,8))

fig.suptitle('Net Wins by League', x=0.045, y=1.01, fontsize=24, fontweight='bold', horizontalalignment='left')



for y, x in zip(lst_years, lst_x):

    # Filter pvt for year

    pvt_net_wins_yx = pvt_net_wins_yl[np.in1d(pvt_net_wins_yl.index.get_level_values(0), y)]

    pvt_net_wins_yx = pvt_net_wins_yx.reset_index(level=0, drop=True)

    

    p = plt.subplot(2,2,x)



    # Plot across Leagues

    pvt_net_wins_yx['b_net_wins'].plot(kind='bar', 

                                       color=pvt_net_wins_yx.positive.map({True:blues[5], 

                                                                           False:reds[5]}))

    

    # Format each plot

    plt.title(y, fontsize= 18, fontweight='bold')

    plt.xlabel('')

    plt.ylabel('Count', fontsize=12)

    plt.ylim(y_min, y_max)



plt.tight_layout()

fig.subplots_adjust(top=0.89)

plt.show()  
dct_seasons = {'Spring_Season':'SPRS', 'Summer_Season':'SUMS', 'Spring_Playoffs':'SPRP', 'Summer_Playoffs':'SUMP',

              'Regional':'REG', 'International':'INT', 'Winter_Season':'WNTRS', 'Winter_Playoffs':'WNTRP'}



# Map Seasons

df['SA'] = df['Season'].map(dct_seasons)

print(pd.unique(df['SA']))

df[['Season', 'SA']].head()
# Pivot by Year and Season

pvt_net_wins_ys = df.pivot_table(index = ['Year', 'SA'], values = ['bResult', 'rResult'], aggfunc=np.sum)

pvt_net_wins_ys['b_net_wins'] = pvt_net_wins_ys['bResult'] - pvt_net_wins_ys['rResult']

pvt_net_wins_ys['positive'] = pvt_net_wins_ys['b_net_wins'] > 0



# Color Formatting

blues = sns.color_palette('Blues')

reds = sns.color_palette('Reds')



lst_years = pd.unique(pvt_net_wins_yl.index.get_level_values(0))

lst_x = [1, 2, 3, 4, 5, 6, 7]



y_max = 1.25 * pvt_net_wins_ys['b_net_wins'].max()

y_min = 1.25 * pvt_net_wins_ys['b_net_wins'].min()



fig, ax = plt.subplots(2,2, figsize = (14,8))

fig.suptitle('Net Wins by Season', x=0.05, y=1.01, fontsize=24, fontweight='bold', horizontalalignment='left')



for y, x in zip(lst_years, lst_x):

    # Filter pvt for year

    pvt_net_wins_yx = pvt_net_wins_ys[np.in1d(pvt_net_wins_ys.index.get_level_values(0), y)]

    pvt_net_wins_yx = pvt_net_wins_yx.reset_index(level=0, drop=True)

    

    p = plt.subplot(2,2,x)



    # Plot across Leagues

    pvt_net_wins_yx['b_net_wins'].plot(kind='bar', 

                                       color=pvt_net_wins_yx.positive.map({True:blues[5], 

                                                                           False:reds[5]}))

    # Format each plot

    plt.title(y, fontsize= 18, fontweight='bold')

    plt.xlabel('')

    plt.ylabel('Count', fontsize=12)

    plt.ylim(y_min, y_max)

    

plt.tight_layout()

fig.subplots_adjust(top=0.89)

plt.show()  
# Adding win team to reformat data

df['win_team'] = np.where(df['bResult']==1, 'blue', 'red')



# Creating factor plot

g = sns.factorplot(x='win_team', col='Season', row='Year', data=df, kind='count', legend_out=True, 

                  palette = {'blue':blues[5], 'red':reds[5]}, margin_titles=True)

plt.show()
df_columns[8:16]
# Another look

df[['gamelength', 'golddiff']].head(1)
print('gamelength value:', df['gamelength'][0])

print('golddiff list length:', len(df['golddiff'][0]))
print('gamelength data type:', type(df['gamelength'][0]))

print('golddiff data type:', type(df['golddiff'][0]))
df[['goldblue','bKills','bTowers', 'bInhibs', 'bDragons', 'bBarons']].head()
# Use literal_eval to convert golddiff to list dat atype

from ast import literal_eval

df['golddiff'] = df['golddiff'].apply(literal_eval)



# Check to make sure function works correctly

print(type(df['golddiff'][0]))

print(df['golddiff'][0])
# Transform all other columns containing pseudo lists to real lists

df['goldblue'] = df['goldblue'].apply(literal_eval)

df['bKills'] = df['bKills'].apply(literal_eval)

df['bTowers'] = df['bTowers'].apply(literal_eval)

df['bInhibs'] = df['bInhibs'].apply(literal_eval)

df['bDragons'] = df['bDragons'].apply(literal_eval)

df['bBarons'] = df['bBarons'].apply(literal_eval)

df['bHeralds'] = df['bHeralds'].apply(literal_eval)



df['goldred'] = df['goldred'].apply(literal_eval)

df['rKills'] = df['rKills'].apply(literal_eval)

df['rTowers'] = df['rTowers'].apply(literal_eval)

df['rInhibs'] = df['rInhibs'].apply(literal_eval)

df['rDragons'] = df['rDragons'].apply(literal_eval)

df['rBarons'] = df['rBarons'].apply(literal_eval)

df['rHeralds'] = df['rHeralds'].apply(literal_eval)



df['goldblueTop'] = df['goldblueTop'].apply(literal_eval)

df['goldblueJungle'] = df['goldblueJungle'].apply(literal_eval)

df['goldblueMiddle'] = df['goldblueMiddle'].apply(literal_eval)

df['goldblueADC'] = df['goldblueADC'].apply(literal_eval)

df['goldblueSupport'] = df['goldblueSupport'].apply(literal_eval)



df['goldredTop'] = df['goldredTop'].apply(literal_eval)

df['goldredJungle'] = df['goldredJungle'].apply(literal_eval)

df['goldredMiddle'] = df['goldredMiddle'].apply(literal_eval)

df['goldredADC'] = df['goldredADC'].apply(literal_eval)

df['goldredSupport'] = df['goldredSupport'].apply(literal_eval)
# Checking the length of a random row in bTowers

print('golddiff:'+str(len(df['golddiff'][1234])))

print('goldblue:'+str(len(df['goldblue'][1234])))

print('goldred:'+str(len(df['goldred'][1234])))
df['gamelength'][1234]
# Create dictionary of games and wins 

dct_win_team = dict(zip(df.index, df['win_team']))
# How the df currently looks for goldblue and goldred

df[['goldblue', 'goldred']].head()
# Unstack df to dfu, reformatting data to have minutes along columns

df_gold = df[['goldblue', 'goldred']].unstack().apply(pd.Series)



# Map Level 1 to dct_win_team

df_gold['win_team'] = (df_gold.index.get_level_values(1))

df_gold['win_team'] = df_gold['win_team'].map(dct_win_team)

df_gold.head()
# Transform "variable" index to column

df_gold = df_gold.reset_index(level=0, drop=False)

df_gold = df_gold.rename(columns={'level_0':'variable'})

df_gold.head()
# Reformat dataframe to transform minute columns to a single column with the minute value within it through pd.melt

melt_gold = pd.melt(df_gold, ['variable', 'win_team'], var_name='minute')

print(melt_gold.shape)

melt_gold.head()
melt_gold['var_color'] = np.where(melt_gold['variable']=='goldblue', 'blue', 'red')

melt_gold['win'] = 'no'

melt_gold.loc[((melt_gold['var_color']=='blue') & (melt_gold['win_team']=='blue')) |

             ((melt_gold['var_color']=='red') & (melt_gold['win_team']=='red')), 

              'win'] = 'yes'

melt_gold.head()
melt_gold.shape
# Check that wins for blue 

melt_gold[(melt_gold['win']=='no') & (melt_gold['win_team']=='blue')].head()
melt_gold[(melt_gold['win']=='no') & (melt_gold['win_team']=='red')].head()
melt_gold.pivot_table(index=['win_team'], values=['win'], aggfunc='count', margins=True)
sns.palplot(sns.color_palette('summer', 20))

colors = sns.color_palette('summer', 20)
# Create pal dictionary using comprehension

pal = {win: colors[2] if win =='yes' else colors[19] for win in melt_gold.win.unique()}



fig, ax = plt.subplots(4,1, figsize=(14,14))

fig.suptitle('Total Gold Value - Win', fontsize=24, fontweight='bold', x=0.06, y=1.025, horizontalalignment='left')

fig.subplots_adjust(top=0.85)



plt.subplot(4,1,1)

p1 = sns.boxplot(x='minute', y='value', hue='win', data = melt_gold[melt_gold['minute']<20], 

                 palette=pal)

box = p1.artists[38] # artists allow you to access the box appearance properties of the box you want to highlight

box.set_facecolor(sns.xkcd_rgb['blue'])

box = p1.artists[39] # artists allow you to access the box appearance properties of the box you want to highlight

box.set_facecolor(sns.xkcd_rgb['red'])

plt.ylabel('Gold Value')



plt.subplot(4,1,2)

ax = sns.boxplot(x='minute', y='value', hue='win', data = melt_gold[(melt_gold['minute']>=20) & ((melt_gold['minute'] <40))],

                palette = pal)

plt.ylabel('Gold Value')



plt.subplot(4,1,3)

sns.boxplot(x='minute', y='value', hue='win', data = melt_gold[(melt_gold['minute']>=40) & ((melt_gold['minute']<60))], 

            palette=pal)

plt.ylabel('Gold Value')



plt.subplot(4,1,4)

sns.boxplot(x='minute', y='value', hue='win', data = melt_gold[(melt_gold['minute']>=60)], 

            palette=pal)

plt.ylabel('Gold Value')



plt.tight_layout()

plt.show()
df[['bKills', 'rKills']].head()
type(df['bKills'][0][0][0])
print(len(df['bKills'][0]))

print(len(df['bKills'][1]))

print(len(df['bKills'][2]))
# Creating function to only extract  minutelist

def extract_minutes(item):

    """Extract minute list from bKills or rKills column"""

    try:

        return[item[0] for item in item]

    except:

        return[0 for item in item] # to account for empty lists
# Apply function to new Kills columns

df['bKills_min'] = df['bKills'].apply(extract_minutes)

df['rKills_min'] = df['rKills'].apply(extract_minutes)

df[['bKills', 'bKills_min', 'rKills', 'rKills_min']].head()
df[['bKills', 'bKills_min', 'rKills', 'rKills_min']].shape
# Check to make sure function applied correctly



# First row, 3rd sublist, 0 position

print(df['bKills'][0][3][0])

# First row, 3rd value

print(df['bKills_min'][0][3])



# 125th row, 10th sublist, 0 position

print(df['bKills'][125][10][0])

# 125th row, 3rd value

print(df['bKills_min'][125][10])
# Identify unique game ID in gameHash

print(df['MatchHistory'][1023])

print(df['MatchHistory'][1023][-16:])
# Extract and gameID from df['MatchHistory]

gameIDs = []



for row in df['MatchHistory']:

    # Take 16 characters from end of MatchHistory and append to gameIDs

    gameIDs.append(row[-16:])



df['gameID'] = gameIDs

df['gameID'] = df['LA'] + '-' + df['gameID']



dct_index_gameID = dict(zip(df.index, df['gameID']))

dct_gameID_win = dict(zip(df['gameID'], df['win_team']))



df[['MatchHistory', 'gameID', 'win_team', 'bKills_min', 'rKills_min']].head()
len(df)
# Unstack df to dfu, reformatting data to have kill number along columns

df_kills = df[['bKills_min', 'rKills_min']].unstack().apply(pd.Series)

df_kills.tail()
# Map Level 1 index to dct_index_gameID and dct_win_team

df_kills['gameID'] = (df_kills.index.get_level_values(1))

df_kills['gameID'] = df_kills['gameID'].map(dct_index_gameID)



df_kills['win_team'] = (df_kills.index.get_level_values(1))

df_kills['win_team'] = df_kills['gameID'].map(dct_gameID_win)



# Transform "variable" index to column

df_kills = df_kills.reset_index(level=0, drop=False)

df_kills = df_kills.rename(columns={'level_0':'variable'})

df_kills.head()
df_kills.shape
# Reformat dataframe to transform columns (kill number) to a single column with the minute value within it through pd.melt

melt_kills = pd.melt(df_kills, ['gameID', 'variable', 'win_team'], var_name='action_count').fillna(0)

melt_kills.head()
# Additional melt_kills columns

melt_kills['var_color'] = np.where(melt_kills['variable']=='bKills_min', 'blue', 'red')



melt_kills['win'] = False

melt_kills.loc[((melt_kills['var_color']=='blue') & (melt_kills['win_team']=='blue')) | 

               ((melt_kills['var_color']=='red') & (melt_kills['win_team']=='red')), 'win'] = True



melt_kills['action_count'] = melt_kills['action_count'] + 1

melt_kills['minute_bin'] = pd.cut(melt_kills['value'], bins=np.arange(0,82,5))

melt_kills.head()
melt_kills[melt_kills['gameID']=='NA-fbb300951ad8327c'][:35]
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

sns.palplot(sns.xkcd_palette(colors))
df_kills = melt_kills[melt_kills['minute_bin'].notnull()]

df_kills_win = df_kills[df_kills['win']==True]

df_kills_lose = df_kills[df_kills['win']==False]



# Setstyle options

sns.set_style('whitegrid')

sns.palplot(sns.color_palette('Greens', 20))

color_win = sns.xkcd_rgb['medium green']

color_lose = 'black'



# Create Figure

fig, ax = plt.subplots(3,2, figsize=(14,10))

fig.suptitle('Kill Distributions', x=0.075, y=1.03, fontsize=24, fontweight='bold', 

             horizontalalignment='left')

fig.subplots_adjust(top=0.9)



percentiles = np.array([25, 50, 75])



# Create Subplots



# ------------------------------------- WIN -------------------------------------



# 1 Box and Whisker

p1 = plt.subplot2grid((3,2), (0,0), colspan=1)

sns.boxplot(y=df_kills_win['action_count'], color=color_win, showfliers=False)

plt.title('Wining Teams', fontsize=18, fontweight='bold')

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel('All Games', fontsize=18)

plt.ylabel('Kills', fontsize = 18)



# 2 ECDF Plot

p2 = plt.subplot2grid((3,2), (1,0), colspan=1)

x = np.sort(df_kills_win['action_count'])

y = np.arange(1, len(x) + 1) / len(x)

ptiles_gl_win = np.percentile(df_kills_win['action_count'], percentiles)

plt.plot(x,y, marker='.', linestyle='none', color=color_win)

plt.plot(ptiles_gl_win, percentiles/100, marker='D', color='red', linestyle='none')



# 2 ECDF Formatting (a lot)

yvals = p2.get_yticks()

p2.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

plt.yticks(fontsize=14)

plt.xlabel('Kills', fontsize=18)

plt.ylabel('ECDF', fontsize=18)

plt.margins(0.02)



# 3 Histogram Count

p3 = plt.subplot2grid((3,2), (2,0), colspan=1)

plt.hist(x='action_count', bins=80, data=df_kills_win, color=color_win)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel('Kills', fontsize = 18)

plt.ylabel('Count of All Games', fontsize=18)



# 3 Histogram Percentage - Second Y Axis for Percent (To DO - align Y2 ytick values to Y1 ytick lines)

weights = np.ones_like(df_kills_win['action_count']) / len(df_kills_win['action_count'])

p3 = plt.twinx()

plt.hist(x='action_count', bins=80, weights= weights, data=df_kills_win, color=color_win)

yvals = p3.get_yticks()

p3.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

p3.grid(b=False)





# ------------------------------------- LOSE -------------------------------------



# 4 Box and Whisker

p4 = plt.subplot2grid((3,2), (0,1), colspan=1)

sns.boxplot(y=df_kills_lose['action_count'], color='grey', showfliers=False)

plt.title('Losing Teams', fontsize=18, fontweight = 'bold')

plt.ylim(0,28)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel('All Games', fontsize=18)

plt.ylabel('Kills', fontsize = 18)



# 5 ECDF Plot

p5 = plt.subplot2grid((3,2), (1,1), colspan=1)

ptiles_gl_lose = np.percentile(df_kills_lose['action_count'], percentiles)

x = np.sort(df_kills_lose['action_count'])

y = np.arange(1, len(x) + 1) / len(x)

plt.plot(x,y, marker='.', linestyle='none', color=color_lose)

plt.plot(ptiles_gl_lose, percentiles/100, marker='D', color='red', linestyle='none')



# 5 ECDF Formatting (a lot)

yvals = p5.get_yticks()

p5.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])



plt.yticks(fontsize=14)

plt.xlim(0, 41)

plt.xlabel('Kills', fontsize=18)

plt.ylabel('ECDF', fontsize=18)

plt.margins(0.02)



# 6 Histogram Count

p6 = plt.subplot2grid((3,2), (2,1), colspan=1)

plt.hist(x='action_count', bins=80, data=df_kills_lose, color='black')

plt.ylim(0,4000)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.xlabel('Kills', fontsize = 18)

plt.ylabel('Count of All Games', fontsize=18)



# 6 Histogram Percentage - Second Y Axis for Percent (To DO - align Y2 ytick values to Y1 ytick lines)

weights = np.ones_like(df_kills_lose['action_count']) / len(df_kills_lose['action_count'])

p6 = plt.twinx()

plt.hist(x='action_count', bins=80, weights= weights, data=df_kills_lose, color=color_lose)

yvals = p6.get_yticks()

p6.set_yticklabels(['{:3.0f}%'.format(y*100) for y in yvals])

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.ylim(0,43)

p6.grid(b=False)



# Show everything

plt.tight_layout()

plt.show()
lst_gameID = pd.unique(df['gameID'])

games = 4

lst_games = list(np.arange(0,games,1))



fig, ax = plt.subplots(figsize=(14,16))

fig.suptitle('Cumulative Kills by Game', fontsize= 24, fontweight='bold', x=0.04, y=1.02, horizontalalignment='left')

fig.subplots_adjust(top=0.85)



for g, c in zip(lst_gameID, lst_games):

    gameID = lst_gameID[c]

    

    ax = plt.subplot(games,1,c+1)



    df_kills = melt_kills[(melt_kills['gameID']==gameID) & melt_kills['minute_bin'].notnull()]



    plt.title('GameID = ' + gameID, fontsize= 14, loc='left')

    sns.regplot(x="value", y="action_count", y_jitter=True, data=df_kills[df_kills['win']==True], color=color_win, 

                label='winning team')

    sns.regplot(x="value", y="action_count", y_jitter=True, data=df_kills[df_kills['win']==False], color=color_lose, 

                label='losing team')

    plt.yticks(np.arange(0, 25, 2))

    plt.xticks(np.arange(0, 45, 1))

    plt.ylabel('Cumulative Kills')

    plt.xlabel('Minute')



    ax.legend(loc='best')



plt.tight_layout()

plt.show()
df_kills = melt_kills[melt_kills['minute_bin'].notnull()]



fig, ax = plt.subplots(figsize=(14,6))

fig.suptitle('Cumulative Kills by Team', fontsize= 24, fontweight='bold', x=0.04, y=1.04, horizontalalignment='left')

# fig.subplots_adjust(top=0.85)



ax = plt.subplot(111)



sns.regplot(x="value", y="action_count", y_jitter=True, data=df_kills[df_kills['win']==True], color=color_win, 

            label='winning team', scatter_kws={'s':2})

sns.regplot(x="value", y="action_count", y_jitter=True, data=df_kills[df_kills['win']==False], color=color_lose, 

            label='losing team', scatter_kws={'s':2})

plt.yticks(np.arange(0, max(df_kills['action_count'])+2, 2))

plt.xticks(np.arange(0, max(df_kills['value'])+2, 2))

plt.ylabel('Cumulative Kills')

plt.xlabel('Minute')



ax.legend(loc='best')



plt.tight_layout()

plt.show()
df_test = df_kills[df_kills['value']<1]

print('Rows:', len(df_test))

print('Unique Games:',len(pd.unique(df_test['gameID'])))

df_test.head()
df_kills[df_kills['gameID']=='EUR-3468d308b6f7a920']
df[['gameID', 'rKills', 'rKills_min']][df['gameID']=='EUR-3468d308b6f7a920']
df['rKills'][889]
df_kills[df_kills['gameID']=='LCK-198df12c6b9433a9']
# Remove kills less than 1 minute.

print('Rows pre 1 minute exlusion:',len(df_kills))

df_kills = df_kills[df_kills['value']>=1]

print('Rows post 1 minute exlusion:',len(df_kills))
df_kills_min_x = df_kills.copy(deep=True)

df_kills_min_x = df_kills_min_x[(df_kills_min_x['value']<=2)]

print(pd.unique(df_kills_min_x['action_count']))

print(len(pd.unique(df_kills_min_x['gameID'])))
df_kills_min_x[df_kills_min_x['action_count']>2]
df_kills_min_x[df_kills_min_x['gameID']=='SWC-763613ed628ef784']
df_kills_min_x[df_kills_min_x['gameID']=='LCK-a4f63fc013228353']
sns.lmplot(x='action_count', y='win', data=df_kills_min_x, 

           y_jitter=.02, logistic=True, scatter_kws={'s':5})

plt.title('First 2 Minutes of Game')

plt.ylabel('Win Probability')

plt.xlabel('Cumulative Kills')

plt.show()
df_kills_min_x = df_kills.copy(deep=True)

df_kills_min_x = df_kills_min_x[(df_kills_min_x['value']<=5)]

print(pd.unique(df_kills_min_x['action_count']))

print(len(pd.unique(df_kills_min_x['gameID'])))



sns.lmplot(x='action_count', y='win', data=df_kills_min_x, 

               y_jitter=.02, logistic=True, scatter_kws={'s':2})

plt.title('First 5 Minutes of Game')

plt.ylabel('Win Probability')

plt.xlabel('Cumulative Kills')

plt.show()
df_kills_min_x.head()
lst_bins = list(pd.unique(df_kills_min_x['minute_bin'])[:5].sort_values())
df_kills_min_x = df_kills.copy(deep=True)

df_kills_min_x['minute_integer'] = df_kills_min_x['value'].astype(int)

df_kills_min_x = df_kills_min_x[(df_kills_min_x['minute_integer']<=16)]



g = sns.lmplot(x='action_count', y='win', col='minute_integer', col_wrap = 4, data=df_kills_min_x, 

               x_jitter=True, y_jitter=.02, logistic=True, scatter_kws={'s':2}, size=3, aspect=1, sharex=False, sharey = False)

g.set_xlabels('Kill Count', fontweight='bold')

g.set_ylabels('Win Probability', fontweight='bold')

g.set(yticks=np.arange(-.1, 1.1, .1))

g.fig.suptitle('Win Probabilities of Team Kills by Minute\nLogistic Regression', horizontalalignment='left', x=0.05, fontsize=24)

g.fig.subplots_adjust(top=0.87)
# Timestamp values in towers data is unsorted.  First create new columns of sorted tower destruction times.

df['bTowers_sorted'] = df.bTowers.sort_values().apply(lambda x: sorted(x))

df['rTowers_sorted'] = df.rTowers.sort_values().apply(lambda x: sorted(x))



# Unstack bTowers_sorted and rTowers_sorted

df_towers = df[['bTowers_sorted', 'rTowers_sorted']].unstack().apply(pd.Series)

df_towers.head()



# Map Level 1 index to dct_index_gameID and dct_win_team

df_towers['gameID'] = (df_towers.index.get_level_values(1))

df_towers['gameID'] = df_towers['gameID'].map(dct_index_gameID)



df_towers['win_team'] = (df_towers.index.get_level_values(1))

df_towers['win_team'] = df_towers['gameID'].map(dct_gameID_win)



# Transform "variable" index to column

df_towers = df_towers.reset_index(level=0, drop=False)

df_towers = df_towers.rename(columns={'level_0':'variable'})



# Reformat dataframe to transform columns (tower number) to a single column with the minute value within it through pd.melt

melt_towers = pd.melt(df_towers, ['gameID', 'variable', 'win_team'], var_name='tower_count').fillna(0)



# Additional melt_towers columns

melt_towers['var_color'] = np.where(melt_towers['variable']=='bTowers_sorted', 'blue', 'red')



melt_towers['win'] = False

melt_towers.loc[((melt_towers['var_color']=='blue') & (melt_towers['win_team']=='blue')) | 

               ((melt_towers['var_color']=='red') & (melt_towers['win_team']=='red')), 'win'] = True

melt_towers.head(10)



melt_towers['tower_count'] = melt_towers['tower_count'] + 1

melt_towers['minute_bin'] = pd.cut(melt_towers['value'], bins=np.arange(0,82,5))

melt_towers['minute_integer'] = melt_towers['value'].astype(int)

df_towers = melt_towers[melt_towers['minute_bin'].notnull()]



df_towers = df_towers.sort_values(by=['gameID', 'variable', 'value'])

df_towers.head(25)
# Quick check to make sure this all worked

# check_gameID = 'NA-055b17da8456fdc8'

check_gameID = 'NA-0ed1cd0e0e57329c'

df_towers[df_towers['gameID']==check_gameID]
print(df['bTowers'][df['gameID']==check_gameID])

print(df['rTowers'][df['gameID']==check_gameID])
print(lst_gameID)
lst_gameID[3]
g = sns.factorplot(x='minute_integer', y='tower_count', hue='win', col='minute_bin', col_wrap = 4, 

                   data=df_towers)

g.set_xticklabels(rotation=90)

g.scatter_kws={'s':2}

g.fig.suptitle('Towers Destroyed by Minute', horizontalalignment='left', fontsize=24, fontweight='bold', x=0.02, y=1.03)

plt.tight_layout()