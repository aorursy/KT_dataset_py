# import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
pd.options.mode.chained_assignment = None  # default='warn'
# clean and merge NGS data for concussion plays
# Read in 2016 NGS
df1 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
df2 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv')
df3 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv')
df4 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv')
df5 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-post.csv')
df1.dropna(subset=['GSISID'], inplace=True) 

# create list of applicable  2016(plays w/ concussions) values
Game_Key_IDs = [5,21,29,45,54,60,144,149,189,218,231,234,266,274,280,280,281,289,296]
Play_IDs = [3129,2587,538,1212,1045,905,2342,3663,3509,3468,1976,3278,2902,3609,2918,3746,1526,2341,2667]
GSISIDS = [31057,29343,31023,33121,32444,30786,32410,28128,27595,28987,32214,28620,23564,23742,32120,27654,28987,32007,32783,32482,31059,31941,28249,31756,
29815,23259,29629,31950,32807,27860,31844,31785,32725,33127,30789,32998,32810]
# set column origination
df1['Origin'] = 'NGS-2016-pre.csv'
df2['Origin'] = 'NGS-2016-reg-wk1-6.csv'
df3['Origin'] = 'NGS-2016-reg-wk7-12.csv'
df4['Origin'] = 'NGS-2016-reg-wk13-17.csv'
df5['Origin'] = 'NGS-2016-post.csv'
# filter NGS datasets to only concussion plays
df1_filtered = df1[df1['GameKey'].isin(Game_Key_IDs) & df1['PlayID'].isin(Play_IDs) & df1['GSISID'].isin(GSISIDS)]
df1_filtered['GSISID'] = df1_filtered['GSISID'].astype('int64')

df2_filtered = df2[df2['GameKey'].isin(Game_Key_IDs) & df2['PlayID'].isin(Play_IDs) & df2['GSISID'].isin(GSISIDS)]
df2_filtered.dropna(subset=['GSISID'], inplace=True)

df3_filtered = df3[df3['GameKey'].isin(Game_Key_IDs) & df3['PlayID'].isin(Play_IDs) & df3['GSISID'].isin(GSISIDS)]
df3_filtered.dropna(subset=['GSISID'], inplace=True) 

df4_filtered = df4[df4['GameKey'].isin(Game_Key_IDs) & df4['PlayID'].isin(Play_IDs) & df4['GSISID'].isin(GSISIDS)]
df4_filtered.dropna(subset=['GSISID'], inplace=True) 

df5_filtered = df5[df5['GameKey'].isin(Game_Key_IDs) & df5['PlayID'].isin(Play_IDs) & df5['GSISID'].isin(GSISIDS)]
df5_filtered.dropna(subset=['GSISID'], inplace=True)
df2_filtered['GSISID'] = df2_filtered['GSISID'].astype('int64')
df3_filtered['GSISID'] = df3_filtered['GSISID'].astype('int64')
df4_filtered['GSISID'] = df4_filtered['GSISID'].astype('int64')
df5_filtered['GSISID'] = df5_filtered['GSISID'].astype('int64')

# Build 2016 NGS dataframe
NGS16_data = pd.concat([df1_filtered, df2_filtered, df3_filtered, df4_filtered, df5_filtered])
# Read in 2017 NGS
df1 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv')
df2 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv')
df3 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv')
df4 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv')
df5 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-post.csv')
# set column origination
df1['Origin'] = 'NGS-2017-pre.csv'
df2['Origin'] = 'NGS-2017-reg-wk1-6.csv'
df3['Origin'] = 'NGS-2017-reg-wk7-12.csv'
df4['Origin'] = 'NGS-2017-reg-wk13-17.csv'
df5['Origin'] = 'NGS-2017-post.csv'

# 2017 applicable (concussion plays) keys
Game_Key_IDs = [357,364,364,384,392,397,399,414,448,473,506,553,567,585,585,601,607,618]
Play_IDs = [3630,2489,2764,183,1088,1526,3312,1262,2792,2072,1988,1683,1407,2208,733,602,978,2792]
GSISIDS = [30171,31313,32323,33813,32615,32894,26035,33941,33838,29492,27060,32820,32403,33069,30384,33260,29793,31950,29384, 32851, 31930,33841,31999,31763,27442,31317,33445,25503,32891,24535,31697,32114,32677]


df1_filtered = df1[df1['GameKey'].isin(Game_Key_IDs) & df1['PlayID'].isin(Play_IDs) & df1['GSISID'].isin(GSISIDS)]
df1_filtered.dropna(subset=['GSISID'], inplace=True) 
df2_filtered = df2[df2['GameKey'].isin(Game_Key_IDs) & df2['PlayID'].isin(Play_IDs) & df2['GSISID'].isin(GSISIDS)]
df2_filtered.dropna(subset=['GSISID'], inplace=True) 
df3_filtered = df3[df3['GameKey'].isin(Game_Key_IDs) & df3['PlayID'].isin(Play_IDs) & df3['GSISID'].isin(GSISIDS)]
df3_filtered.dropna(subset=['GSISID'], inplace=True) 
df4_filtered = df4[df4['GameKey'].isin(Game_Key_IDs) & df4['PlayID'].isin(Play_IDs) & df4['GSISID'].isin(GSISIDS)]
df4_filtered.dropna(subset=['GSISID'], inplace=True)
df5_filtered = df5[df5['GameKey'].isin(Game_Key_IDs) & df5['PlayID'].isin(Play_IDs) & df5['GSISID'].isin(GSISIDS)]
df5_filtered.dropna(subset=['GSISID'], inplace=True)

df1_filtered['GSISID'] = df1_filtered['GSISID'].astype('int64')
df2_filtered['GSISID'] = df2_filtered['GSISID'].astype('int64')
df3_filtered['GSISID'] = df3_filtered['GSISID'].astype('int64')
df4_filtered['GSISID'] = df4_filtered['GSISID'].astype('int64')
df5_filtered['GSISID'] = df5_filtered['GSISID'].astype('int64')
# 2017 NGS dataframe
NGS17_data = pd.concat([df1_filtered, df2_filtered, df3_filtered, df4_filtered, df5_filtered])
# release memory
df1 = []
df2 = []
df3 = []
df4 = []
df5 = []

NGS_All = pd.concat([NGS16_data,NGS17_data])
# write to CSV
NGS_All.to_csv('NGS_All.csv',header=True)
NGS_All.columns = [col.lower() for col in NGS_All.columns]
# read in concussion plays from video review dataset to build unique key list
vr = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
vr1 = vr[['Season_Year','GameKey','PlayID','GSISID']]
vr2 = vr[['Season_Year','GameKey','PlayID','Primary_Partner_GSISID']]
vr2.rename(columns={'Primary_Partner_GSISID': 'GSISID'}, inplace=True)
vr_final = pd.concat([vr1,vr2])
vr_final= vr_final.drop_duplicates(['Season_Year','GameKey','PlayID', 'GSISID'])
unique_grp_ids = vr_final['Season_Year'].astype(str) + vr_final['GameKey'].astype(str) + vr_final['PlayID'].astype(str) + vr_final['GSISID'].astype(str)
# Create a Natural Key to use across datasets and in group by statements - sort the dataframe
NGS_All['groupid'] = NGS_All['season_year'].astype(str) + NGS_All['gamekey'].astype(str) + NGS_All['playid'].astype(str) + NGS_All['gsisid'].astype(str)
NGS_All = NGS_All.groupby('groupid').apply(lambda x: x.sort_values('time'))
# remove the multi index created by the above
NGS_All = NGS_All.reset_index(0, drop=True)
# filter to only gameids within the video review dataset, record count down to 21604 from 33722
NGS_All = NGS_All[NGS_All['groupid'].isin(unique_grp_ids)]
# identify the Start and End of each play according to Event (ball_snap and tackle)
play_start = NGS_All.loc[NGS_All['event']=='ball_snap',['time','groupid']]
play_start.rename(columns={'time': 'play_start'}, inplace=True)
play_end = NGS_All.loc[NGS_All['event']=='tackle',['time','groupid']]
play_end.rename(columns={'time': 'play_end'}, inplace=True)

play_run_times = pd.merge(play_start,play_end,on='groupid',how='inner')
# join NGS_all to play_start and create a play_start column
NGS_All = pd.merge(NGS_All, play_run_times, on='groupid', how='outer')
# reduce dataframe to only data after ball is snapped and when the play is blown dead
ngs = NGS_All[(NGS_All['time']>= NGS_All['play_start']) & (NGS_All['time'] <= NGS_All['play_end'])]
# only 45 groupids in the NextGen data that match to video review data out of possible 
ngs['groupid'].unique().size
# Calculate common attributes like distnace & kinetics - speed, acceleration, instaneous velocity, etc

# create distance in meters, 1 yard = .9144 meters
ngs['distance'] = ngs['dis'] * 0.9144
# calculate total distance (yards and meters)
yd_distance_ttl = ngs.groupby(['groupid'])['dis'].sum()
mt_distance_ttl = ngs.groupby(['groupid'])['distance'].sum()
yd = yd_distance_ttl.to_frame().reset_index()
m = mt_distance_ttl.to_frame().reset_index()

# begin creation of a Fact Table
fact_tbl = pd.merge(yd, m, on='groupid')
fact_tbl= fact_tbl.rename(columns={'dis': 'yd_ttl', 'distance': 'meter_ttl'})

# to datetime
ngs['time'] = pd.to_datetime(ngs['time'])
# get total time by play
time_diff = ngs.groupby('groupid')['time'].apply(lambda x: x.max() - x.min())
# change to a dataframe
time_diff = time_diff.to_frame().reset_index()
# merge onto fact table
fact_tbl = pd.merge(fact_tbl,time_diff, on='groupid')
# convert into seconds
fact_tbl['time'] = fact_tbl['time'].dt.seconds
fact_tbl['avg_speed_m'] = fact_tbl['meter_ttl'] / fact_tbl['time']
fact_tbl['avg_speed_yd'] = fact_tbl['yd_ttl'] / fact_tbl['time']
# calc Instantaneous velocity (meters) at every decisecond
ngs['instant_velocity_m'] = ngs['distance'] / .1
ngs = ngs.reset_index(drop=True)
#store previous velocity
ngs['prev_velocity'] = ngs.groupby(['groupid'])['instant_velocity_m'].shift(1)
# if first record of group, set initial velocity to 0
ngs['prev_velocity'].fillna(0, inplace=True)
# calculate instantaneous acceleration
ngs['instant_acceleration'] = ((ngs.instant_velocity_m - ngs.prev_velocity) / .1)
# View the comparative speed and distance traveled of the players involved in a concussion
fact_tbl[fact_tbl['groupid'].str.contains("2016144")]
# add additional keys to the fact table
vr = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
vr.columns = [col.lower() for col in vr.columns]
additional_keys=vr.drop_duplicates(['gamekey','playid','gsisid','primary_partner_gsisid'])
ak = additional_keys[['season_year','gamekey','playid','gsisid','player_activity_derived','primary_impact_type','primary_partner_gsisid']]
ak2 = additional_keys[['season_year','gamekey','playid','player_activity_derived','primary_impact_type','primary_partner_gsisid']]
ak['groupid'] = additional_keys['season_year'].astype(str) + additional_keys['gamekey'].astype(str) + additional_keys['playid'].astype(str) + additional_keys['gsisid'].astype(str)
ak2['groupid'] = additional_keys['season_year'].astype(str) + additional_keys['gamekey'].astype(str) + additional_keys['playid'].astype(str) + additional_keys['primary_partner_gsisid'].astype(str)
ak2['gsisid'] = ak2['primary_partner_gsisid']
#concat full additional keys, clean data
ak3 = pd.concat([ak, ak2])
ak3 = ak3.dropna(axis='rows')
ak3 = ak3[ak3.gsisid != 'Unclear']
final_fact = pd.merge(fact_tbl,ak3,on='groupid')
# define player type for easier visualization, c=concussed, p=primary partner
final_fact['player_type'] = np.where((final_fact.gsisid == final_fact.primary_partner_gsisid), 'p', 'c')
import seaborn as sns
sns.set(color_codes=True)

# split by concussed or primary player impacted
concussed = final_fact[final_fact['player_type']=='c'][:]
primary = final_fact[final_fact['player_type']=='p'][:]
data = final_fact
sns.set(style="darkgrid")
g = sns.catplot(x="player_activity_derived", hue="player_type", col="primary_impact_type",
                data=data, kind="count");
# view average distance traveled (yds) & average speed (yds/sec) for concussed and primary partner

f, axes = plt.subplots(1, 2, figsize=(10,5))
sns.distplot(concussed['yd_ttl'], label="Concussed", kde=False, ax=axes[0])
sns.distplot(primary['yd_ttl'], label="Primary", kde=False, ax=axes[0])
sns.distplot(concussed['avg_speed_yd'],label="Concussed", kde=False, ax=axes[1])
sns.distplot(primary['avg_speed_yd'], label = "Primary", kde=False, ax=axes[1])
plt.legend(loc=(1.04,.5))
# look at distribution of speed as a function of distance (yards) for concussed players, where do most instances
# occur across those variables? 
sns.set(style="ticks")

x = concussed['yd_ttl']
y = concussed['avg_speed_yd']
x_primary = primary['yd_ttl']
y_primary = primary['avg_speed_yd']


c = sns.jointplot(x, y, data=concussed, kind="scatter")
p = sns.jointplot(x_primary, y_primary, data=concussed, kind="scatter", color='r')
# deeper look into the plays where distance traveled is >40yds per the earlier distribution pattern
viz = final_fact[['groupid','player_type','season_year','gamekey','playid']][final_fact['yd_ttl']>40]
viz1 = viz.groupby(['season_year','gamekey','playid']).size().reset_index(name='counts')
viz1 = viz1[viz1['counts']>1]
viz1 = viz1.reset_index()

# add player type to ngs data to make visualizations simpler
ngs = pd.merge(ngs, final_fact[['player_type','groupid']], on='groupid', how='inner')

# turn into a loop to dynamically draw plots
import matplotlib.patches as mpatches
for index, row in viz1.iterrows():
    groupid = row['season_year'].astype(str) + row['gamekey'].astype(str) + row['playid'].astype(str)
    player_concussed = ngs[ngs['groupid'].str.contains(groupid)] 
    
    x= []
    y = []
    x_primary = []
    y_primary = []
        
    
    #for every row in player_concussed
    for i, r in player_concussed.iterrows():
        
        if r['player_type']=='c':
            x.append(r['x'])
            y.append(r['y'])
        
        else:
            x_primary.append(r['x'])
            y_primary.append(r['y'])
    
    
    plt.figure(figsize=(6,3))
    plt.title("Year: " + row['season_year'].astype(str) + " GameKey: " + row['gamekey'].astype(str) 
             + " Play Id: " + row['playid'].astype(str))
    plt.scatter(x, y, s=2, c="b");
    plt.xlabel("Field Length (yd)")
    plt.ylabel("Field Width (yd)")
    plt.scatter(x_primary, y_primary, s=2, c="r")
    blue_patch = mpatches.Patch(color='blue', label='Concussed Player')
    red_patch = mpatches.Patch(color='red', label='Primary Player')
    plt.legend(handles=[red_patch,blue_patch])
    
# plot starting point for concussed and primary players
starting_positions = ngs[(ngs['event']=='ball_snap')]
starting_positions = starting_positions[['x','y','groupid']]
starting_positions = starting_positions.rename(columns={'x': 'x_start', 'y': 'y_start'})
# plot ending position for concussed and primary players (not entirely accurate as it is not the exact point of impact
# where tracking data ends)
ending_positions = ngs[(ngs['event']=='tackle')]
ending_positions = ending_positions[['x','y','groupid']]
ending_positions = ending_positions.rename(columns={'x': 'x_end', 'y': 'y_end'})
final_fact = pd.merge(final_fact,ending_positions,on='groupid',how='inner')
final_fact = pd.merge(final_fact,starting_positions,on='groupid',how='inner')
# calculate distance (yards) traveled height and width-wise of field
final_fact['x_traveled'] = abs(final_fact['x_end'] - final_fact['x_start'])
final_fact['y_traveled'] = abs(final_fact['y_end'] - final_fact['y_start'])
final_fact.head(5)
img = plt.imread("../input/football-fieldpng/football_field.png")
plt.scatter(final_fact['x_end'], final_fact['y_end'], c="r")
plt.imshow(img, zorder=0, extent=[0, 100, 0, 53])
plt.show()
concussed_players = final_fact[final_fact['player_type']=='c']
sns.jointplot(x="x_end", y="y_end", data=final_fact, kind="kde");
# y (width) location on field where concussion occured
sns.distplot(concussed_players['y_end']);
len(concussed_players[concussed_players['y_end'].between(43, 53, inclusive=True)]) + len(concussed_players[concussed_players['y_end'].between(0, 10, inclusive=True)])
## Observe all NFL Punt Plays
punts = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
punts.columns = [col.lower() for col in punts.columns]
# create a key
punts['key'] = punts['season_year'].astype(str) + punts['gamekey'].astype(str) + punts['playid'].astype(str)
# modify description to be a string for parsing
punts['playdescription'] = punts['playdescription'].astype(str)

# search for fair catch punts
import re
fair_catch = []
for row in punts['playdescription']:
    match = re.search('fair catch', row)
    if match:
        fair_catch.append(1)
    elif match is None:
        fair_catch.append(0)
        
punts["fair_catch"] = fair_catch   

#create lists for concussion play keys
gamekeys = concussed_players['gamekey']
playids = concussed_players['playid']
punts2 = punts[punts['fair_catch']==1]

fair_catch_concussed = punts2[punts2['gamekey'].isin(gamekeys) & punts2['playid'].isin(playids)]
# Only 1 fair catch resulted in a concussion
# .53% of all punnt plays result in concussions
((vr.gamekey.count() -1 )/ punts.gamekey.count() * 100)

#Rule proposals have been submitted via the assosciated presentation.
