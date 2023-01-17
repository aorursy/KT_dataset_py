import numpy as np

import pandas as pd

import math

import statistics

import random



%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



# Convert to float32 or int16 to save memory usage

play_dtype = {'PlayDay': 'int16', 'PlayGame': 'int16', 'Temperature': 'float32'}



trk_dtype = {'time': 'float32', 'x': 'float32', 'y': 'float32', 

        'dis': 'float32', 's': 'float32', 'o': 'float32', 'dir': 'float32'}



inj = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

play = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv", dtype=play_dtype)

trk = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv", dtype=trk_dtype)
inj.head()
print('inj null')

print(inj.isnull().sum())
inj[inj['GameID'].duplicated(keep=False)]
# Convert the period of injuries into 4-grade.(M1: 0.25, M7: 0.5, M28: 0.75, M42: 1)

def GetInjuryGrade(x):

    if x['DM_M42'] == 1:

        return 1

    elif x['DM_M42'] == 0 and x['DM_M28'] == 1:

        return 0.75

    elif x['DM_M28'] == 0 and x['DM_M7'] == 1:

        return 0.5

    elif x['DM_M7'] == 0 and x['DM_M1'] == 1:

        return 0.25    

inj['GetInjuryGrade'] = inj.apply(GetInjuryGrade, axis=1)

print('InjuryGrade Mean: {0:.3f}'.format(inj['GetInjuryGrade'].mean()))
play.head()
print('play null')

print(play.isnull().sum())
InjPlayers = inj['PlayerKey'].unique()

NoInjPlayers = play.loc[~play['PlayerKey'].isin(InjPlayers)]['PlayerKey'].unique()

print('InjPlayers: ', len(InjPlayers))

print('NoInjPlayers: ', len(NoInjPlayers))
StadiumType_dict = {

    'Outdoor': 'outdoor', 'Outdoors': 'outdoor', 'Indoors': 'indoor', 'Dome': 'indoor', 

    'Retractable Roof': 'indoor', 'Indoor': 'indoor', 'Open': 'outdoor', 

    'Domed, closed': 'indoor', 'Retr. Roof - Closed': 'indoor', 

    'Retr. Roof-Closed': 'indoor', 'Domed, open': 'outdoor', 

    'Dome, closed': 'indoor', 'Closed Dome': 'indoor', 

    'Domed': 'indoor', 'Oudoor': 'outdoor', 'Domed, Open': 'outdoor', 

    'Ourdoor': 'outdoor', 'Outdoor Retr Roof-Open': 'outdoor', 'Outddors': 'outdoor', 

    'Indoor, Roof Closed': 'indoor', 'Retr. Roof-Open': 'outdoor', 

    'Retr. Roof - Open': 'outdoor', 'Indoor, Open Roof': 'outdoor', 'Bowl': 'outdoor', 

    'Retr. Roof Closed': 'indoor', 'Heinz Field': 'outdoor', 'Outdor': 'outdoor', 

    'Outside': 'outdoor', 'Cloudy': 'outdoor', np.nan: np.nan

}



play['StadiumType'] = play['StadiumType'].apply(lambda x: StadiumType_dict[x])





Weather_dict = {

    'Cloudy': 'overcast', 'Sunny': 'clear', 'Partly Cloudy': 'overcast', 'Clear': 'clear', 

    'Mostly Cloudy': 'overcast', 'Rain': 'rain', 'Controlled Climate': np.nan, 

    'N/A (Indoors)': np.nan, 'Indoors': np.nan, 'Mostly Sunny': 'clear',  'Indoor': np.nan, 

    'Partly Sunny': 'clear', 'Mostly cloudy': 'clear', 

    'Fair': 'clear', 'N/A Indoor': np.nan, 

    'Light Rain': 'rain', 'Partly cloudy': 'overcast', 'Clear and warm': 'clear', 

    'Mostly sunny': 'clear', 'Hazy': 'overcast', 'cloudy': 'overcast', 

    'Snow': 'snow', 'Overcast': 'overcast', 

    'Clear Skies': 'clear', 'Cloudy and Cool': 'overcast', 'Clear skies': 'clear', 

    'Cloudy, 50% change of rain': 'rain', 

    'Cloudy, fog started developing in 2nd quarter': 'overcast', 

    'Clear and cold': 'clear', 

    'Partly clear': 'clear', 'Cloudy and cold': 'overcast', 

    'Sunny and clear': 'clear', 'Rain Chance 40%': 'overcast', 

    'Sunny and warm': 'clear', 'Clear and Cool': 'clear', 

    'Sunny, highs to upper 80s': 'clear', 'Sunny Skies': 'clear', 

    'Cloudy, light snow accumulating 1-3"': 'snow', 'Scattered Showers': 'rain', 

    'Clear and Sunny': 'clear', 'Mostly Coudy': 'overcast', 

    'Rain likely, temps in low 40s.': 'rain', 'Cold': 'overcast', 

    'Sunny and cold': 'clear', 'Partly sunny': 'clear', 'Showers': 'rain', 

    'Rainy': 'rain', 'Clear to Partly Cloudy': 'clear', 

    'Clear and sunny': 'clear', 'Sunny, Windy': 'clear', 'Rain shower': 'rain', 

    'Cloudy, chance of rain': 'overcast', 'Heat Index 95': 'clear', 

    'Mostly Sunny Skies': 'clear', '10% Chance of Rain': 'overcast', 

    'Sun & clouds': 'clear', 'Cloudy, Rain': 'rain', 'Heavy lake effect snow': 'snow', 

    '30% Chance of Rain': 'overcast', 'Partly Clouidy': 'overcast', 

    'Coudy': 'overcast', 

    'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.': 'rain', 

    'Party Cloudy': 'overcast', np.nan: np.nan

}   



play['Weather'] = play['Weather'].apply(lambda x: Weather_dict[x])





PlayTypeDict = {

    'Pass': 'Pass', 

    'Rush': 'Rush', 

    'Extra Point': 'Extra Point', 

    'Kickoff': 'Kickoff', 

    'Punt': 'Punt', 

    'Field Goal': 'Field Goal', 

    'Kickoff Not Returned': 'Kickoff', 

    'Punt Not Returned': 'Punt', 

    'Kickoff Returned': 'Kickoff', 

    'Punt Returned': 'Punt', 

    '0': np.nan, 

    np.nan: np.nan

}



play['PlayType'] = play['PlayType'].apply(lambda x: PlayTypeDict[x])





play.loc[play['PositionGroup'] == 'Missing Data', 'PositionGroup'] = np.nan
play.loc[play['StadiumType'] != 'indoor']['Temperature'].value_counts().sort_index()
# Temperature unknown is sometimes -999.0. It should be missing value.

play.loc[play['Temperature'] == -999.0, 'Temperature'] = np.nan

outdoor_temp = play.loc[(play['StadiumType'] == 'outdoor')&(play['Temperature'].notnull())].drop_duplicates(subset='GameID')['Temperature']

temp_q25, temp_q50, temp_q75 = np.percentile(outdoor_temp, [25, 50, 75])

TemperatureMean = outdoor_temp.mean()

TemperatureStd = outdoor_temp.std()

outdoor_temp.hist(bins=75)

print('Temperature outdoor\nMean: {0:.3f}\nStd: {1:.3f}'.format(TemperatureMean, TemperatureStd))

print('Quartile \n25th: {}\n75th: {}'.format(temp_q25, temp_q75))
game_num = play['GameID'].unique().shape[0]

outdoor_natural = play[(play['StadiumType'] == 'outdoor')&(play['FieldType'] == 'Natural')]['GameID'].unique().shape[0]

indoor_natural = play[(play['StadiumType'] == 'indoor')&(play['FieldType'] == 'Natural')]['GameID'].unique().shape[0]

outdoor_synthetic = play[(play['StadiumType'] == 'outdoor')&(play['FieldType'] == 'Synthetic')]['GameID'].unique().shape[0]

indoor_synthetic = play[(play['StadiumType'] == 'indoor')&(play['FieldType'] == 'Synthetic')]['GameID'].unique().shape[0]



fig = plt.figure(figsize=(6, 4))

ax1 = fig.add_subplot(111)

ax1.bar(

    range(4), np.array([outdoor_natural, indoor_natural, outdoor_synthetic, indoor_synthetic])/game_num*100, 

    tick_label=['out_nat', 'in_nat', 'out_syn', 'in_syn'], color='darkgreen', width=0.8, alpha=0.5)

ax1.set_xlabel('StadiumType/FieldType')

ax1.set_ylabel('Ratio(%)')



fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 20,

    'color': 'black'

}

for i, stadium_field_type in enumerate([outdoor_natural, indoor_natural, outdoor_synthetic, indoor_synthetic]):

    ax1.text(i, 10, '{0:.1f}%'.format(stadium_field_type/game_num*100), alpha=0.8, horizontalalignment='center', fontdict=fps)

ax1.set_title('Stadium/Field type ratio', 

              fontdict={'family': 'sans-serif', 'weight': 'bold','size': 18,}

             )

plt.show()
fig = plt.figure(figsize=(4, 4))



ax1 = fig.add_subplot(111)

ax1.pie(

    play['PlayType'].value_counts().sort_values(ascending=False).values, 

    labels = play['PlayType'].value_counts().sort_values(ascending=False).index, 

    labeldistance = 1.1, 

    textprops = {'fontsize': 12}, 

)

ax1.set_title('Play type ratio', 

              fontdict={'family': 'sans-serif', 'weight': 'bold','size': 18,}

             )

plt.show()
inj_detailed = pd.merge(inj, play[['GameID', 'PlayerDay', 'FieldType', 'StadiumType', 'Temperature', 'Weather', 'Position', 'PositionGroup', 'RosterPosition']].drop_duplicates(subset='GameID'), on='GameID', how='left')

inj_detailed = pd.merge(inj_detailed, play[['PlayKey', 'PlayerGamePlay']], on='PlayKey', how='left')

inj_detailed.drop(columns=['Surface'], inplace=True)

inj_detailed.loc[inj_detailed['PlayKey'].notnull(), 'PlayType'] = inj_detailed.loc[inj_detailed['PlayKey'].notnull()].apply(lambda x: play.loc[play['PlayKey']==x['PlayKey'], 'PlayType'].iloc[0], axis=1)
# Injury probablity per PlayType

PlayTypeCount = play.groupby('PlayType')['PlayerKey'].count().reset_index()

PlayTypeCount.columns = ['PlayType', 'PlayTypeCount']

InjPlayTypeCount = inj_detailed.groupby('PlayType')['PlayerKey'].count().reset_index()

InjPlayTypeCount.columns = ['PlayType', 'InjPlayTypeCount']

InjPRPlayType = pd.merge(PlayTypeCount, InjPlayTypeCount, on='PlayType', how='left')

InjPRPlayType['PRPlayType'] = (InjPRPlayType['InjPlayTypeCount'] / InjPRPlayType['PlayTypeCount'])*100

InjPRPlayType['PRPlayType'].fillna(0, inplace=True)
fig = plt.figure(figsize=(6, 4))



ax1 = fig.add_subplot(111)

ax1.bar(np.arange(0, len(InjPRPlayType)*2, 2), InjPRPlayType['PRPlayType'], tick_label=InjPRPlayType['PlayType'], color='yellow', width=1, alpha=0.75)

ax1.set_xlabel('PlayType')

ax1.set_ylabel('Probablity(%)')

ax1.set_ylim([0, 0.12])

fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 12,

    'color': 'darkred'

}

ax1.text(0, 0.09, 'Punt and Kickoff play \nincrease the risk of injury?', alpha=0.75, fontdict=fps)

ax1.grid(False)

ax1.set_title('Injury probablity by play type', 

              fontdict={'family': 'sans-serif', 'weight': 'bold','size': 18,}

             )

plt.show()
# Injury probablity per PositionGroup

PosCount = play.groupby('PositionGroup')['PlayerKey'].count().reset_index()

PosCount.columns = ['PositionGroup', 'PosCount']

InjPosCount = inj_detailed.groupby('PositionGroup')['PlayerKey'].count().reset_index()

InjPosCount.columns = ['PositionGroup', 'InjPosCount']

InjPRPos = pd.merge(PosCount, InjPosCount, on='PositionGroup', how='left')

InjPRPos['PRPos'] = (InjPRPos['InjPosCount'] / InjPRPos['PosCount'])*100

InjPRPos['PRPos'].fillna(0, inplace=True)
fig = plt.figure(figsize=(6, 4))



ax1 = fig.add_subplot(111)

ax1.bar(np.arange(0, len(InjPRPos)*2, 2), InjPRPos['PRPos'], tick_label=InjPRPos['PositionGroup'], color='yellow', width=1, alpha=0.75)

ax1.set_xlabel('PositionGroup')

ax1.set_ylabel('Probablity(%)')

#ax1.set_ylim([0, 0.12])

fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 12,

    'color': 'darkred'

}

ax1.text(0, 0.06, 'RB and TE \nincrease the risk of injury?', alpha=1, fontdict=fps)

ax1.grid(False)

ax1.set_title('Injury probablity by position group', 

              fontdict={'family': 'sans-serif', 'weight': 'bold','size': 18,}

             )

plt.show()
print('trk null')

print(trk.isnull().sum())

NoTrkPKs = play.loc[~play['PlayKey'].isin(trk['PlayKey'].unique())]['PlayKey'].unique()

print(len(NoTrkPKs), 'playkeys have no track data.')

NoTrkPKs = list(NoTrkPKs)
trk['event'].value_counts()
# Plays start at some events (ball_snap, snap_direct, kickoff, free_kick)

# Movement before a play start such as huddle might not be related to injuries.

# Plays not including these event should be excluded.

play_start = ['ball_snap', 'snap_direct', 'kickoff', 'onside_kick', 'free_kick']



def GetNoPlayStart(d):

    if (d['event'].isin(play_start)).any():

        return False

    else:

        return True



play_start_series = trk.groupby('PlayKey').apply(GetNoPlayStart)

no_play_start_pks = play_start_series[play_start_series==True].index

trk = trk[~trk['PlayKey'].isin(no_play_start_pks)]

NoTrkPKs += list(no_play_start_pks)
LowTemp, HighTemp = temp_q25, temp_q75





InjPKs = list(inj_detailed.loc[inj_detailed['PlayKey'].notnull()]['PlayKey'].unique())

InjNoPKsGameID = list(inj_detailed.loc[inj_detailed['PlayKey'].isnull()]['GameID'].unique())

InjNoPKs = list(play.loc[play['GameID'].isin(InjNoPKsGameID)]['PlayKey'].unique())



InjPKsGoodCond = list(inj_detailed.loc[

    (inj_detailed['PlayKey'].notnull())

    &(~play['PlayKey'].isin(NoTrkPKs))

    &((inj_detailed['StadiumType']=='indoor')

    |((inj_detailed['Weather'].isin(['clear', 'overcast']))

    &(inj_detailed['Temperature'] <= HighTemp)

    &(inj_detailed['Temperature'] >= LowTemp)))

]['PlayKey'].unique())



NoInjPKs = list(play.loc[(~play['PlayKey'].isin(InjPKs+InjNoPKs))]['PlayKey'].unique())



NoInjPKsGoodCond = list(play.loc[

    (play['PlayKey'].isin(NoInjPKs))

    &(~play['PlayKey'].isin(NoTrkPKs))

    &((play['StadiumType']=='indoor')

    |((play['Weather'].isin(['clear', 'overcast']))

    &(play['Temperature'] <= HighTemp)

    &(play['Temperature'] >= LowTemp)))

]['PlayKey'].unique())





SampNoInjPKsGoodCond = random.sample(list(NoInjPKsGoodCond), 10000)
def GetMaxSpeed(d):

    d['distance'] = np.sqrt(

        (d['x'] - d['x'].shift(1))**2 + (d['y'] - d['y'].shift(1))**2

    )

    d['speed'] = d['distance'] / (d['time'] - d['time'].shift(1))

    play_d = d[

        d['time'] >= d.loc[d['event'].isin(play_start), 'time'].min() - 3.0

    ]

    return pd.Series(play_d['speed'].max())
InjMaxSpeed = trk[trk['PlayKey'].isin(InjPKsGoodCond)].groupby('PlayKey').apply(GetMaxSpeed).reset_index(level='PlayKey')

InjMaxSpeed.columns = ['PlayKey', 'MaxSpeed']

inj_mvt = pd.merge(inj_detailed, InjMaxSpeed, on='PlayKey', how='left')



NoInjMaxSpeed = trk[trk['PlayKey'].isin(SampNoInjPKsGoodCond)].groupby('PlayKey').apply(GetMaxSpeed).reset_index(level='PlayKey')

NoInjMaxSpeed.columns = ['PlayKey', 'MaxSpeed']

noinj_mvt = pd.merge(play, NoInjMaxSpeed, on='PlayKey', how='left')



# Fix abnormal values

inj_mvt.loc[inj_mvt['MaxSpeed'] > 10, 'MaxSpeed'] = 10

noinj_mvt.loc[noinj_mvt['MaxSpeed'] > 10, 'MaxSpeed'] = 10
fig = plt.figure(figsize=(12, 10))

grid = plt.GridSpec(3, 2, hspace=0.5, wspace=0.1)



ax1 = fig.add_subplot(grid[0, :])

hist1 = inj_mvt.loc[inj_mvt['MaxSpeed'].notnull()]['MaxSpeed']

hist2 = noinj_mvt.loc[noinj_mvt['MaxSpeed'].notnull()]['MaxSpeed']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax1.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='Inj')

ax1.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax1.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='InjMedian')

ax1.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax1.annotate('', xy=(axvline1, 0.2), size=15, xytext=(axvline2, 0.2), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



handles, labels = ax1.get_legend_handles_labels()

handles[0], handles[1], handles[2], handles[3] = handles[1], handles[0], handles[3], handles[2]

labels[0], labels[1], labels[2], labels[3] = labels[1], labels[0], labels[3], labels[2]

ax1.legend(handles[::-1], labels[::-1], ncol=2, bbox_to_anchor=(1, 1))



ax1.set_xlabel('MaxSpeed(y/s)')

ax1.set_ylabel('Percent(%)')



fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 16,

    'color': 'darkred'

}

ax1.text(0, 0.25, 'Injured players had \nhigher max speed \n(median: {} > {})'.format(round(axvline1, 2), round(axvline2, 2)), alpha=0.75, fontdict=fps)



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax1.set_ylim([0, 0.4])

ax1.set_title('Injury vs NoInjury', fontdict=font)





ax2 = fig.add_subplot(grid[1, :])

hist1 = inj_mvt.loc[(inj_mvt['MaxSpeed'].notnull())&(inj_mvt['BodyPart']=='Knee')]['MaxSpeed']

hist2 = noinj_mvt.loc[noinj_mvt['MaxSpeed'].notnull()]['MaxSpeed']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)





ax2.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax2.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax2.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax2.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax2.annotate('', xy=(axvline1, 0.2), size=15, xytext=(axvline2, 0.2), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 16,

    'color': 'darkred'

}

ax2.text(0, 0.25, 'High max speed increase \nknee injury risk? ', alpha=0.75, fontdict=fps)



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax2.set_ylim([0, 0.4])

ax2.set_title('KneeInjury vs NoInjury', fontdict=font)





ax3 = fig.add_subplot(grid[2, 0])

hist1 = inj_mvt.loc[(inj_mvt['MaxSpeed'].notnull())&(inj_mvt['FieldType']=='Natural')]['MaxSpeed']

hist2 = noinj_mvt.loc[(noinj_mvt['MaxSpeed'].notnull())&(noinj_mvt['FieldType']=='Natural')]['MaxSpeed']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax3.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax3.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax3.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax3.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax3.set_ylim([0, 0.4])

ax3.set_title('Injury vs NoInjury (OnNatTurf)', fontdict=font)





ax4 = fig.add_subplot(grid[2, 1])

hist1 = inj_mvt.loc[(inj_mvt['MaxSpeed'].notnull())&(inj_mvt['FieldType']=='Synthetic')]['MaxSpeed']

hist2 = noinj_mvt.loc[(noinj_mvt['MaxSpeed'].notnull())&(noinj_mvt['FieldType']=='Synthetic')]['MaxSpeed']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax4.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax4.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax4.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax4.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax4.set_ylim([0, 0.4])

ax4.set_title('Injury vs NoInjury (OnSynTurf)', fontdict=font)





fig.suptitle('Max Speed', fontweight='bold', size=24)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# Direction change degree per 1/10th of a second mean

def GetDirChg(d):

    d['dir_chg'] = abs(d['dir'] - d['dir'].shift(1))

    d.loc[d['dir_chg'] > 180, 'dir_chg'] = 360 - d['dir_chg']

    play_d = d[

        d['time'] >= d.loc[d['event'].isin(play_start), 'time'].min() - 3.0

    ] 

    return play_d['dir_chg'].mean()
InjDirChg = trk[trk['PlayKey'].isin(InjPKsGoodCond)].groupby('PlayKey').apply(GetDirChg).reset_index(level='PlayKey')

InjDirChg.columns = ['PlayKey', 'DirChg']

inj_mvt = pd.merge(inj_mvt, InjDirChg, on='PlayKey', how='left')



NoInjDirChg = trk[trk['PlayKey'].isin(SampNoInjPKsGoodCond)].groupby('PlayKey').apply(GetDirChg).reset_index(level='PlayKey')

NoInjDirChg.columns = ['PlayKey', 'DirChg']

noinj_mvt = pd.merge(noinj_mvt, NoInjDirChg, on='PlayKey', how='left')
fig = plt.figure(figsize=(12, 10))

grid = plt.GridSpec(3, 2, hspace=0.5, wspace=0.1)



ax1 = fig.add_subplot(grid[0, :])

hist1 = inj_mvt.loc[inj_mvt['DirChg'].notnull()]['DirChg']

hist2 = noinj_mvt.loc[noinj_mvt['DirChg'].notnull()]['DirChg']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax1.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='Inj')

ax1.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax1.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='InjMedian')

ax1.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax1.annotate('', xy=(axvline1, 0.2), size=15, xytext=(axvline2, 0.2), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



handles, labels = ax1.get_legend_handles_labels()

handles[0], handles[1], handles[2], handles[3] = handles[1], handles[0], handles[3], handles[2]

labels[0], labels[1], labels[2], labels[3] = labels[1], labels[0], labels[3], labels[2]

ax1.legend(handles[::-1], labels[::-1], ncol=2, bbox_to_anchor=(1, 1))



ax1.set_xlabel('DirChg(deg)')

ax1.set_ylabel('Percent(%)')



fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 16,

    'color': 'darkred'

}

ax1.text(0, 0.25, 'Injured players changed \nmore direction \n(median: {} > {})'.format(round(axvline1, 2), round(axvline2, 2)), alpha=0.75, fontdict=fps)



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax1.set_ylim([0, 0.4])

ax1.set_title('Injury vs NoInjury', fontdict=font)





ax2 = fig.add_subplot(grid[1, 0])

hist1 = inj_mvt.loc[(inj_mvt['DirChg'].notnull())&(inj_mvt['BodyPart']=='Knee')]['DirChg']

hist2 = noinj_mvt.loc[noinj_mvt['DirChg'].notnull()]['DirChg']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax2.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax2.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax2.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax2.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax2.annotate('', xy=(axvline1, 0.2), size=15, xytext=(axvline2, 0.2), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax2.set_title('KneeInjury vs NoInjury', fontdict=font)

ax2.set_ylim([0, 0.4])





ax3 = fig.add_subplot(grid[1, 1])

hist1 = inj_mvt.loc[(inj_mvt['DirChg'].notnull())&(inj_mvt['BodyPart']=='Ankle')]['DirChg']

hist2 = noinj_mvt.loc[noinj_mvt['DirChg'].notnull()]['DirChg']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax3.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='AnkleInj')

ax3.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax3.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='AnkleInjMedian')

ax3.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax3.annotate('', xy=(axvline1, 0.2), size=15, xytext=(axvline2, 0.2), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax3.set_title('AnkleInjury vs NoInjury', fontdict=font)

ax3.set_ylim([0, 0.4])





ax4 = fig.add_subplot(grid[2, 0])

hist1 = inj_mvt.loc[(inj_mvt['DirChg'].notnull())&(inj_mvt['FieldType']=='Natural')&(inj_mvt['BodyPart']=='Ankle')]['DirChg']

hist2 = noinj_mvt.loc[(noinj_mvt['DirChg'].notnull())&(noinj_mvt['FieldType']=='Natural')]['DirChg']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax4.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax4.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax4.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax4.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax4.set_ylim([0, 0.4])

ax4.set_title('Injury vs NoInjury (OnNatTurf)', fontdict=font)





ax5 = fig.add_subplot(grid[2, 1])

hist1 = inj_mvt.loc[(inj_mvt['DirChg'].notnull())&(inj_mvt['FieldType']=='Synthetic')&(inj_mvt['BodyPart']=='Ankle')]['DirChg']

hist2 = noinj_mvt.loc[(noinj_mvt['DirChg'].notnull())&(noinj_mvt['FieldType']=='Synthetic')]['DirChg']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax5.hist(hist1, bins=25, range=(0, 12), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax5.hist(hist2, bins=100, range=(0, 12), density=True, color='green', alpha=0.5, label='NoInj')

ax5.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax5.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax5.set_ylim([0, 0.4])

ax5.set_title('Injury vs NoInjury (OnSynTurf)', fontdict=font)





fig.suptitle('Direction change per a tenth of a second', fontweight='bold', size=24)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
eg_pk = '46074-7-26'

eg_trk = trk[trk['PlayKey'] == eg_pk]

eg_trk['distance'] = np.sqrt(

    (eg_trk['x'] - eg_trk['x'].shift(1))**2 + (eg_trk['y'] - eg_trk['y'].shift(1))**2

)

eg_trk['speed'] = eg_trk['distance'] / (eg_trk['time'] - eg_trk['time'].shift(1))

eg_trk['accdec'] = (eg_trk['speed'] - eg_trk['speed'].shift(1)) / (eg_trk['time'] - eg_trk['time'].shift(1))

play_eg_trk = eg_trk[

    eg_trk['time'] >= eg_trk.loc[eg_trk['event'].isin(play_start), 'time'].min() - 3.0

]
play_eg_trk['accdec'].reset_index(drop=True).plot()

plt.show()
f = play_eg_trk.loc[play_eg_trk['accdec'].notnull()]['accdec'].values

F = abs(np.fft.fft(f)/(len(f)/2))  # normalization

F = F[:int((len(f)/2))]
plt.plot(range(len(F)), F)

plt.show()
def GetAccDec(d):

    d['distance'] = np.sqrt(

        (d['x'] - d['x'].shift(1))**2 + (d['y'] - d['y'].shift(1))**2

    )

    d['speed'] = d['distance'] / (d['time'] - d['time'].shift(1))

    d['accdec'] = (d['speed'] - d['speed'].shift(1)) / (d['time'] - d['time'].shift(1))

    play_d = d[

        d['time'] >= d.loc[d['event'].isin(play_start), 'time'].min() - 3.0

    ]

    f = play_d.loc[play_d['accdec'].notnull()]['accdec'].to_list()

    F = abs(np.fft.fft(f)/(len(f)/2))

    F = F[:int((len(f)/2))]

    return np.sum([math.log(i+1)*n for i, n in enumerate(F)])
InjAccDec = trk[trk['PlayKey'].isin(InjPKsGoodCond)].groupby('PlayKey').apply(GetAccDec).reset_index(level='PlayKey')

InjAccDec.columns = ['PlayKey', 'AccDec']

inj_mvt = pd.merge(inj_mvt, InjAccDec, on='PlayKey', how='left')



NoInjAccDec = trk[trk['PlayKey'].isin(SampNoInjPKsGoodCond)].groupby('PlayKey').apply(GetAccDec).reset_index(level='PlayKey')

NoInjAccDec.columns = ['PlayKey', 'AccDec']

noinj_mvt = pd.merge(noinj_mvt, NoInjAccDec, on='PlayKey', how='left')
fig = plt.figure(figsize=(12, 10))

grid = plt.GridSpec(3, 2, hspace=0.5, wspace=0.1)



ax1 = fig.add_subplot(grid[0, :])

hist1 = inj_mvt.loc[inj_mvt['AccDec'].notnull()]['AccDec']

hist2 = noinj_mvt.loc[noinj_mvt['AccDec'].notnull()]['AccDec']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax1.hist(hist1, bins=25, range=(0, 300), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='Inj')

ax1.hist(hist2, bins=100, range=(0, 300), density=True, color='green', alpha=0.5, label='NoInj')

ax1.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='InjMedian')

ax1.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax1.annotate('', xy=(axvline1, 0.01), size=15, xytext=(axvline2, 0.01), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



handles, labels = ax1.get_legend_handles_labels()

handles[0], handles[1], handles[2], handles[3] = handles[1], handles[0], handles[3], handles[2]

labels[0], labels[1], labels[2], labels[3] = labels[1], labels[0], labels[3], labels[2]

ax1.legend(handles[::-1], labels[::-1], ncol=2, bbox_to_anchor=(1, 1))



ax1.set_xlabel('AccDec')

ax1.set_ylabel('Percent(%)')



fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 16,

    'color': 'darkred'

}

ax1.text(200, 0.005, 'Injured players changed \nspeed frecuently? \n(median: {} > {})'.format(round(axvline1, 2), round(axvline2, 2)), alpha=0.75, fontdict=fps)



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax1.set_ylim([0, 0.025])

ax1.set_title('Injury vs NoInjury', fontdict=font)





ax2 = fig.add_subplot(grid[1, 0])

hist1 = inj_mvt.loc[(inj_mvt['AccDec'].notnull())&(inj_mvt['BodyPart']=='Knee')]['AccDec']

hist2 = noinj_mvt.loc[noinj_mvt['AccDec'].notnull()]['AccDec']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax2.hist(hist1, bins=25, range=(0, 300), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax2.hist(hist2, bins=100, range=(0, 300), density=True, color='green', alpha=0.5, label='NoInj')

ax2.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax2.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax2.set_ylim([0, 0.025])

ax2.set_title('KneeInjury vs NoInjury', fontdict=font)





ax3 = fig.add_subplot(grid[1, 1])

hist1 = inj_mvt.loc[(inj_mvt['AccDec'].notnull())&(inj_mvt['BodyPart']=='Ankle')]['AccDec']

hist2 = noinj_mvt.loc[noinj_mvt['AccDec'].notnull()]['AccDec']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax3.hist(hist1, bins=25, range=(0, 300), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='AnkleInj')

ax3.hist(hist2, bins=100, range=(0, 300), density=True, color='green', alpha=0.5, label='NoInj')

ax3.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='AnkleInjMedian')

ax3.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax3.set_ylim([0, 0.025])

ax3.set_title('AnkleInjury vs NoInjury', fontdict=font)







ax4 = fig.add_subplot(grid[2, 0])

hist1 = inj_mvt.loc[(inj_mvt['AccDec'].notnull())&(inj_mvt['FieldType']=='Natural')]['AccDec']

hist2 = noinj_mvt.loc[(noinj_mvt['AccDec'].notnull())&(noinj_mvt['FieldType']=='Natural')]['AccDec']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax4.hist(hist1, bins=25, range=(0, 300), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax4.hist(hist2, bins=100, range=(0, 300), density=True, color='green', alpha=0.5, label='NoInj')

ax4.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax4.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax4.set_ylim([0, 0.025])

ax4.set_title('Injury vs NoInjury (OnNatTurf)', fontdict=font)





ax5 = fig.add_subplot(grid[2, 1])

hist1 = inj_mvt.loc[(inj_mvt['AccDec'].notnull())&(inj_mvt['FieldType']=='Synthetic')]['AccDec']

hist2 = noinj_mvt.loc[(noinj_mvt['AccDec'].notnull())&(noinj_mvt['FieldType']=='Synthetic')]['AccDec']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax5.hist(hist1, bins=25, range=(0, 300), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax5.hist(hist2, bins=100, range=(0, 300), density=True, color='green', alpha=0.5, label='NoInj')

ax5.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax5.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax5.set_ylim([0, 0.025])

ax5.set_title('Injury vs NoInjury (OnSynTurf)', fontdict=font)





fig.suptitle('Acc/Dec frecuency by Fourier transform', fontweight='bold', size=24)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
def GetDistance(d):

    d['Distance'] = np.sqrt(

        (d['x'] - d['x'].shift(1))**2 + (d['y'] - d['y'].shift(1))**2

    )

    play_d = d[

        d['time'] >= d.loc[d['event'].isin(play_start), 'time'].min() - 3.0

    ]

    return play_d['Distance'].sum()
InjDistance = trk[trk['PlayKey'].isin(InjPKsGoodCond)].groupby('PlayKey').apply(GetDistance).reset_index(level='PlayKey')

InjDistance.columns = ['PlayKey', 'Distance']

inj_mvt = pd.merge(inj_mvt, InjDistance, on='PlayKey', how='left')



NoInjDistance = trk[trk['PlayKey'].isin(SampNoInjPKsGoodCond)].groupby('PlayKey').apply(GetDistance).reset_index(level='PlayKey')

NoInjDistance.columns = ['PlayKey', 'Distance']

noinj_mvt = pd.merge(noinj_mvt, NoInjDistance, on='PlayKey', how='left')
fig = plt.figure(figsize=(12, 10))

grid = plt.GridSpec(3, 2, hspace=0.5, wspace=0.1)

ax1 = fig.add_subplot(grid[0, :])

hist1 = inj_mvt.loc[inj_mvt['Distance'].notnull()]['Distance']

hist2 = noinj_mvt.loc[noinj_mvt['Distance'].notnull()]['Distance']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax1.hist(hist1, bins=25, range=(0, 150), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='Inj')

ax1.hist(hist2, bins=100, range=(0, 150), density=True, color='green', alpha=0.5, label='NoInj')

ax1.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='InjMedian')

ax1.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')

ax1.annotate('', xy=(axvline1, 0.02), size=15, xytext=(axvline2, 0.02), 

             arrowprops=dict(arrowstyle='simple', color='darkred'))



handles, labels = ax1.get_legend_handles_labels()

handles[0], handles[1], handles[2], handles[3] = handles[1], handles[0], handles[3], handles[2]

labels[0], labels[1], labels[2], labels[3] = labels[1], labels[0], labels[3], labels[2]

ax1.legend(handles[::-1], labels[::-1], ncol=2, bbox_to_anchor=(1, 1))



ax1.set_xlabel('AccDec')

ax1.set_ylabel('Percent(%)')



fps = {

    'family': 'monospace', 

    'weight': 'heavy', 

    'size': 16,

    'color': 'darkred'

}

#ax1.text(0, 0.25, 'Injuries changed \nmore direction \n(median: {} > {})'.format(round(axvline1, 2), round(axvline2, 2)), alpha=0.75, fontdict=fps)



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax1.set_ylim([0, 0.04])

ax1.set_title('Injury vs NoInjury', fontdict=font)





ax2 = fig.add_subplot(grid[1, 0])

hist1 = inj_mvt.loc[(inj_mvt['Distance'].notnull())&(inj_mvt['BodyPart']=='Knee')]['Distance']

hist2 = noinj_mvt.loc[noinj_mvt['Distance'].notnull()]['Distance']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax2.hist(hist1, bins=25, range=(0, 150), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax2.hist(hist2, bins=100, range=(0, 150), density=True, color='green', alpha=0.5, label='NoInj')

ax2.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax2.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax2.set_ylim([0, 0.04])

ax2.set_title('KneeInjury vs NoInjury', fontdict=font)





ax3 = fig.add_subplot(grid[1, 1])

hist1 = inj_mvt.loc[(inj_mvt['Distance'].notnull())&(inj_mvt['BodyPart']=='Ankle')]['Distance']

hist2 = noinj_mvt.loc[noinj_mvt['Distance'].notnull()]['Distance']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax3.hist(hist1, bins=25, range=(0, 150), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='AnkleInj')

ax3.hist(hist2, bins=100, range=(0, 150), density=True, color='green', alpha=0.5, label='NoInj')

ax3.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='AnkleInjMedian')

ax3.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 16,

        }

ax3.set_ylim([0, 0.04])

ax3.set_title('AnkleInjury vs NoInjury', fontdict=font)





ax4 = fig.add_subplot(grid[2, 0])

hist1 = inj_mvt.loc[(inj_mvt['Distance'].notnull())&(inj_mvt['FieldType']=='Natural')]['Distance']

hist2 = noinj_mvt.loc[(noinj_mvt['Distance'].notnull())&(noinj_mvt['FieldType']=='Natural')]['Distance']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax4.hist(hist1, bins=25, range=(0, 150), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax4.hist(hist2, bins=100, range=(0, 150), density=True, color='green', alpha=0.5, label='NoInj')

ax4.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax4.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax4.set_ylim([0, 0.04])

ax4.set_title('Injury vs NoInjury (OnNatTurf)', fontdict=font)





ax5 = fig.add_subplot(grid[2, 1])

hist1 = inj_mvt.loc[(inj_mvt['Distance'].notnull())&(inj_mvt['FieldType']=='Synthetic')]['Distance']

hist2 = noinj_mvt.loc[(noinj_mvt['Distance'].notnull())&(noinj_mvt['FieldType']=='Synthetic')]['Distance']

axvline1 = statistics.median(hist1)

axvline2 = statistics.median(hist2)



ax5.hist(hist1, bins=25, range=(0, 150), density=True, color='royalblue', alpha=0.75, linewidth=0.5, label='KneeInj')

ax5.hist(hist2, bins=100, range=(0, 150), density=True, color='green', alpha=0.5, label='NoInj')

ax5.axvline(axvline1, color='red', linestyle='dotted', linewidth=1.5, label='KneeInjMedian')

ax5.axvline(axvline2, color='black', linestyle='dotted', linewidth=1.5, label='NoInjMedian')



ax5.set_ylim([0, 0.04])

ax5.set_title('Injury vs NoInjury (OnSynTurf)', fontdict=font)





fig.suptitle('Distance per play', fontweight='bold', size=24)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
mvt_corr = noinj_mvt[['MaxSpeed', 'DirChg', 'AccDec', 'Distance']].corr()

fig, ax = plt.subplots(figsize=(5, 4)) 

sns.heatmap(mvt_corr, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

plt.show()
fig = plt.figure(figsize=(8, 8))



maxspeed_mean_play = noinj_mvt.groupby('PlayType')['MaxSpeed'].mean()

dirchg_mean_play = noinj_mvt.groupby('PlayType')['DirChg'].mean()



ax1 = fig.add_subplot(211)

ax1.bar(np.arange(0.5, len(maxspeed_mean_play)*2, 2), maxspeed_mean_play, width=0.5, label='MaxSpeed', alpha=0.8)

ax1.bar(np.arange(1.0, len(dirchg_mean_play)*2, 2), dirchg_mean_play, width=0.5, label='DirChg', alpha=0.8)

ax1.set_xticks(np.arange(1, len(InjPRPlayType)*2, 2))

ax1.set_xticklabels(maxspeed_mean_play.index)

ax1.set_ylabel('MaxSpeed(y/s) / DirChg(deg)')

ax1.set_ylim([0, 10])

ax1.legend(loc=2)



ax2 = ax1.twinx()

ax2.bar(np.arange(1.5, len(dirchg_mean_play)*2, 2), InjPRPlayType['PRPlayType'], width=0.5, label='Probablity', color='yellow', alpha=0.5)

ax2.grid(False)

ax2.set_ylabel('Probablity(%)')

ax2.set_ylim([0, 0.12])

ax2.legend(loc=1)





maxspeed_mean_pos = noinj_mvt.groupby('PositionGroup')['MaxSpeed'].mean()

dirchg_mean_pos = noinj_mvt.groupby('PositionGroup')['DirChg'].mean()



ax3 = fig.add_subplot(212)

ax3.bar(np.arange(0.5, len(maxspeed_mean_pos)*2, 2), maxspeed_mean_pos, width=0.5, label='MaxSpeed', alpha=0.8)

ax3.bar(np.arange(1.0, len(dirchg_mean_pos)*2, 2), dirchg_mean_pos, width=0.5, label='DirChg', alpha=0.8)

ax3.set_xticks(np.arange(1, len(maxspeed_mean_pos)*2, 2))

ax3.set_xticklabels(maxspeed_mean_pos.index)

ax3.set_ylabel('MaxSpeed(y/s) / DirChg(deg)')

ax3.set_ylim([0, 8])

ax3.legend(loc=2)



ax4 = ax3.twinx()

ax4.bar(np.arange(1.5, len(InjPRPos)*2, 2), InjPRPos['PRPos'], width=0.5, label='Probablity', color='yellow', alpha=0.5)

ax4.grid(False)

ax4.set_ylabel('Probablity(%)')

ax4.set_ylim([0, 0.1])

ax4.legend(loc=1)

plt.show()