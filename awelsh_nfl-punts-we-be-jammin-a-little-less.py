"""
Created on Sat Jan  5 13:40:42 2019

@author: Andrew Welsh

See my powerpoint presentation for a full explanation of the analsys
This notebook is more my 'scratchpad' to show my work than the final presentation
"""
import pandas as pd
import numpy as np
import tqdm
import gc
import scipy.stats as scs
import matplotlib.pyplot as plt

path ='../input/NFL-Punt-Analytics-Competition/'
print('Load data, read CSVs')
#game level data - stadium, teams, week, location info
print('Read Game data')
gamedata = pd.read_csv(path + 'game_data.csv', encoding='ISO-8859-1')
stadium_type = gamedata[['GameKey','StadiumType','Turf']]

print('Read video review concussion data')
#specific players and play IDs when concussions happened on punts
#I copied the video_review.csv in order to clean the one instance of "Unclear" in the GSISID column
#for this analysis, nulls and values of 'Unclear' are essentially the same, so to keep the datatype 
#clean, I replaced the 'Unclear' value with a null in the dataset
path2 = '../input/video-review-cleaned/'
concussion_events = pd.read_csv(path2 + 'video_review.csv', encoding='ISO-8859-1')
# add flag column for indicator of concussions
concussion_events['conc_flag'] = 1
concussion_events.drop(['Season_Year'], axis=1, inplace=True)
#impute dummy value of 1 for GSISID for the nulls
concussion_events = concussion_events.fillna({'Primary_Partner_GSISID':1})
#force data type of GSISID for later merges
concussion_events['Primary_Partner_GSISID'].astype('float32', inplace=True)

print('Read Player Role data')
#the role/position the players were playing on punt plays; independent of their regular roles
punt_role = pd.read_csv(path + 'play_player_role_data.csv', encoding='ISO-8859-1')
punt_role.drop(['Season_Year'], axis=1, inplace=True)

print('Read all punt play data')
#all punt plays
all_plays = pd.read_csv(path + 'play_information.csv', encoding='ISO-8859-1')
all_plays.drop(['Season_Year'], axis=1, inplace=True)
#find plays that were 'no play', i.e. dead ball penalites, timeouts, or end of quarter
all_plays['no_play'] = all_plays['PlayDescription'].str.contains('No Play', regex=True)
#find touchdowns, touchbacks
all_plays['touchdown'] = all_plays['PlayDescription'].str.contains('TOUCHDOWN', regex=True)
all_plays['blocked'] = all_plays['PlayDescription'].str.contains('BLOCKED', regex=True)
all_plays['touchback'] = all_plays['PlayDescription'].str.contains('Touchback', regex=True)
all_plays['penalty'] = all_plays['PlayDescription'].str.contains('PENALTY', regex=True)
all_plays['fair_catch'] = all_plays['PlayDescription'].str.contains('fair catch', regex=True)

#list of removed plays, so they can be removed from NGS data
removed_plays0 = pd.DataFrame(all_plays[all_plays['no_play'] == True])
removed_plays = removed_plays0[['GameKey','PlayID','no_play']]
#remove no-plays from all punt play data
all_plays = all_plays[all_plays['no_play'] == False]
print('Data cleaned, read NGS files')


#%%
#all NGS data, will be concatenated together.

#force datatypes for NGS files, mostly float32 to save memory
dtypes = {'Season_Year': 'int16',
         'GameKey': 'int16',
         'PlayID': 'int16',
         'GSISID': 'float32',
         'Time': 'str',
         'x': 'float32',
         'y': 'float32',
         'dis': 'float32',
         'o': 'float32',
         'dir': 'float32',
         'Event': 'str'}

col_names = list(dtypes.keys())

ngslist = ['NGS-2016-pre.csv',
           'NGS-2016-reg-wk1-6.csv',
           'NGS-2016-reg-wk7-12.csv',
           'NGS-2016-reg-wk13-17.csv',
           'NGS-2016-post.csv',
           'NGS-2017-pre.csv',
           'NGS-2017-reg-wk1-6.csv',
           'NGS-2017-reg-wk7-12.csv',
           'NGS-2017-reg-wk13-17.csv',
           'NGS-2017-post.csv']

print('Read NGS data')
#initialize df list object
ngsdf_list = []

for i in tqdm.tqdm(ngslist):
    df = pd.read_csv(path+i, usecols=col_names, dtype=dtypes)
    
    ngsdf_list.append(df)

#concatenate the NGS CSVs into one file
ngs0 = pd.concat(ngsdf_list)
print('NGS data ready')
#clean up memory usage
del ngsdf_list
del df
gc.collect()

ngs0['Time'] = pd.to_datetime(ngs0['Time'], format='%Y-%m-%d %H:%M:%S')

#converting yards/sec to mph is 3600 / 1760 = 2.04545... 
#and since each row in NGS data is 0.1s, multiply 2.04545 by 10 = 20.4545
ngs0['mph'] = ngs0['dis']*20.4545
    
#add removed_play indicator column to NGS dataset
ngs = pd.merge(ngs0, removed_plays, how='left', on=['GameKey','PlayID'], copy=True)

#clean up memory usage
del ngs0
gc.collect()

print('Remove non-plays from NGS data')
#remove no-play plays from NGS before summary stats are calculated
ngs = ngs[ngs['no_play'] != False]


#gather max and mean speed for player during concussion events
print('Merge NGS data with concussion event data')
ngs_injuries_player = pd.merge(concussion_events, ngs, how='left', on=['GameKey','PlayID','GSISID'])
ngs_injury_speed_player = ngs_injuries_player.groupby(['GameKey','PlayID','GSISID'], as_index = False)['mph'].agg(
                {'max_mph_player': max,
                 'avg_mph_player': np.mean})

#reusing temp1 here
temp1 = pd.merge(ngs_injury_speed_player, concussion_events, how='inner', on=['GameKey','PlayID','GSISID'], copy=True)
temp1 = pd.merge(temp1, punt_role, how='left', on=['GameKey','PlayID','GSISID'], copy=True)


#rename the Primary_Partner_GSISID column to GSISID to make join work properly
#for whatever reason, joining on the mismatched-names didn't work.. so in the interest of it 
#functioning, I went with this clumsy hack approach
concussion_events.columns=['GameKey',
                           'PlayID',
                           'GSISID_player',
                           'Player_Activity_Derived',
                           'Turnover_Related',
                           'Primary_Impact_Type',
                           'GSISID',
                           'Primary_Partner_Activity_Derived',
                           'Friendly_Fire',
                           'conc_flag']

ngs_injuries_partner = pd.merge(concussion_events, ngs, how='left', on=['GameKey','PlayID','GSISID'], copy=True)

ngs_injury_speed_partner = ngs_injuries_partner.groupby(['GameKey','PlayID','GSISID'], as_index = False)['mph'].agg(
                {'max_mph_partner': max,
                 'avg_mph_partner': np.mean})

temp2 = pd.merge(ngs_injury_speed_partner, concussion_events, how='inner', on=['GameKey','PlayID','GSISID'])
#get partner positional data
temp2 = pd.merge(temp2, punt_role, how='left', on=['GameKey','PlayID','GSISID'], copy=True)

temp3 = temp2[['GameKey','PlayID','max_mph_partner','avg_mph_partner','Role']]
temp3.columns = ['GameKey','PlayID','max_mph_partner','avg_mph_partner','partner_role']

#return column names to original 
concussion_events.columns=['GameKey',
                           'PlayID',
                           'GSISID',
                           'Player_Activity_Derived',
                           'Turnover_Related',
                           'Primary_Impact_Type',
                           'Primary_Partner_GSISID',
                           'Primary_Partner_Activity_Derived',
                           'Friendly_Fire',
                           'conc_flag']

concussion_events = pd.merge(temp1, temp3, on=['GameKey','PlayID'], copy=True)
concussion_events = concussion_events.fillna({'partner_role':'NA'})


#get average, average of the max, and max speeds by punt play role
print('Get average velocities of all punt plays by position')
#need to aggregate the max and avg MPH by player first
punt_role_ngs0 = ngs.groupby(['GameKey','PlayID','GSISID'], as_index = False)['mph'].agg(
                {'max_mph': max,
                 'avg_mph': np.mean})

#create temp df to calculate global stats by role
punt_role_ngs0 = pd.merge(punt_role_ngs0, punt_role, how='left', on=['GameKey','PlayID','GSISID'])

#take average of max, max of max, and overall average by role
punt_role_ngs_max = punt_role_ngs0.groupby(['Role'], as_index = False)['max_mph'].agg(
                {'max_mph_max': max,
                 'max_mph_avg': np.mean})

punt_role_ngs_avg = punt_role_ngs0.groupby(['Role'], as_index = False)['avg_mph'].agg(
                {'count','mean'}).reset_index()

#clean up temp df
del punt_role_ngs0
gc.collect()

#summary statistics, by role, for all plays in dataset
punt_role_ngs = pd.merge(punt_role_ngs_max, punt_role_ngs_avg, how='outer', on=['Role'])

#merge concussion events data to role data
#combine player roles on punt plays with overall punt play data, reusing temp1
temp1 = pd.merge(punt_role, all_plays, how='left', on=['GameKey','PlayID'])

#combine stadium type and turf type data to dataset, to later test whether these are correlated to concussions
temp2 = pd.merge(temp1, stadium_type, how='left', on=['GameKey'])

#join concussion event data to the dataset, knowing that for this specific dataset, there are no instances of 
#concussions occurring to 2 players on the same play
punts = pd.merge(temp2, concussion_events, how='left', on=['GameKey','PlayID','GSISID'])

punts = punts.fillna({'conc_flag':0})

#punts['Turf'].value_counts() shows the data is dirty
#Grass                    51013
#Natural Grass            27124
#Field Turf               13253
#Artificial               12927
#FieldTurf                12149
#UBU Speed Series-S5-M     9079
#DD GrassMaster            4418
#A-Turf Titan              4283
#UBU Sports Speed S5-M     3717
#Natural grass             2436
#FieldTurf 360             1406
#Artifical                  812
#Natural                    790
#grass                      593
#Natrual Grass              462
#Natural Grass              440
#FieldTurf360               396
#UBU Speed Series S5-M      396
#Field turf                 264
#Synthetic                  263
#Naturall Grass             154
# so we're going to recode it

#decare grass_ind column, to create indicator of natural grass vs not
punts['grass_ind']=punts['Turf']

#define "grass_recode" dictionary based on value_counts from raw data
grass_recode = {'Grass':1,
                'Natural Grass':1,
                'Field Turf':0,
                'Artificial':0,
                'FieldTurf':0,
                'UBU Speed Series-S5-M':0,
                'DD GrassMaster':0,
                'A-Turf Titan':0,
                'UBU Sports Speed S5-M':0,
                'Natural grass':1,
                'FieldTurf 360':0,
                'Artifical':0,
                'Natural':1,
                'grass':1,
                'Natrual Grass':1,
                'Natural Grass ':1,
                'Natural Grass':1,
                'FieldTurf360':0,
                'UBU Speed Series S5-M':0,
                'Field turf':0,
                'Synthetic':0,
                'Naturall Grass':1
               }
#replace values in grade recode column with values defined in the dictionary array above
punts = punts.replace(dict(grass_ind=grass_recode))
punts['grass_ind'].astype('float16')

print('Test whether natural grass vs turf is related to concussion rate')
temp1 = punts.groupby(['grass_ind'])['conc_flag'].agg({'count','sum'}).reset_index()
temp1['ratio']= temp1['sum']/temp1['count']
print('Concussions by Grass (or not): \n',temp1)
#no difference based on turf type. There were two helmet-to-ground concussions in the dataset, too few to draw conclusions


print('Prepare formation data')
#get counts of players in various positions
role_count = pd.pivot_table(punt_role, index=['GameKey', 'PlayID'], columns=['Role'], 
                       aggfunc=lambda x: len(x.unique()))['GSISID'].fillna(0)

role_count.reset_index(inplace=True)
#sum of all positional columns, to get count of players encoded
role_count['num_players'] = role_count.iloc[:,2:53].sum(axis=1)
#leave only typical extra-man and one-man-short scenarios, for both teams (20 to 24 players)
role_count1 = role_count.loc[(role_count['num_players']>=20) & (role_count['num_players']<=24)]

#knowing there weren't any plays with 2 concussions, copy of just gamekey and playID will be unique identifier
concussion_flag = concussion_events[['GameKey', 'PlayID','conc_flag']]
all_plays_subset = all_plays[['GameKey', 'PlayID','touchdown','blocked','touchback','penalty','fair_catch']]

role_count1 = pd.merge(role_count1, concussion_flag, how='left', on=['GameKey', 'PlayID'])
role_count1 = role_count1.fillna({'conc_flag':0})
role_count1['outer_sum'] = (
        role_count1['VL'] + role_count1['VLi'] + role_count1['VLo'] +
        role_count1['VR'] + role_count1['VRi'] + role_count1['VRo']
        )
role_count1['outer_sum_grp'] = pd.cut(role_count1['outer_sum'], bins=[-1,2,3,6], labels=['0 to 2','3','4 or more'])

role_count1['interior_side_overload'] = abs(
        (role_count1['PDL1'] + role_count1['PDL2'] + role_count1['PDL3'] + 
         role_count1['PDL4'] + role_count1['PDL5'] + role_count1['PDL6'] + 
         role_count1['PLL'])
        -
        (role_count1['PDR1'] + role_count1['PDR2'] + role_count1['PDR3'] + 
         role_count1['PDR4'] + role_count1['PDR5'] + role_count1['PDR6'] + 
         role_count1['PLR'])
        )
role_count1['interior_side_overload_grp'] = pd.cut(role_count1['interior_side_overload'], bins=[-1,1,10], labels=['0 or 1','2 or more'])

role_count1['middle_sum'] = role_count1['PLR'] + role_count1['PLM'] + role_count1['PLL'] + role_count1['PFB']
role_count1['middle_sum_grp'] = pd.cut(role_count1['middle_sum'], bins=[-1,1,10], labels=['0 or 1','2 or more'])

#offense interior are players protecting the punter; hypothesis is punting teams expecting only 
#return team to focus on return play, and deliver only a token punt block attempt, will have 
#fewer interior players to protect the punter (PC, PPR, PLW, PRW)
role_count1['offense_interior'] = role_count1['PC'] + role_count1['PPR'] + role_count1['PLW'] + role_count1['PRW']
role_count1['offense_interior_grp'] = pd.cut(role_count1['offense_interior'], bins=[-1,2,10], labels=['0 to 2','3 or more'])

role_count1 = pd.merge(role_count1,all_plays_subset, how='inner', on=['GameKey','PlayID'])

#
#
#data cleaning and prep complete
#
#
#
#
#
# count number of jammers. Typical formations are 2, 3 or 4 defenders on gunners
#
#
#
conc_by_role1 = pd.pivot_table(role_count1, index=['outer_sum_grp'], columns=['conc_flag'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
#run this next line, uncommented, if you want all possible values:
#conc_by_role1 = pd.pivot_table(role_count1, index=['outer_sum'], columns=['conc_flag'],
#                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
conc_by_role1.reset_index(inplace=True)
conc_by_role1.columns = ['outer_sum','no_conc','conc']
conc_by_role1['total'] = conc_by_role1['no_conc'] + conc_by_role1['conc']
conc_by_role1['rate'] = conc_by_role1['conc'] / conc_by_role1['total']

#chi-square test of difference in concussion counts by # of jammers
chi2, p, dof, expected = scs.chi2_contingency(conc_by_role1[['no_conc','conc']])
print('p-value, concussion vs non-concussions by # of jammers')
print(p)

#
#
# count number of linebacker/fullback defenders. Possibilities are 0 through 4.
#
#
#

conc_by_role2 = pd.pivot_table(role_count1, index=['middle_sum_grp'], columns=['conc_flag'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
#run this if you want all possible values
#conc_by_role2 = pd.pivot_table(role_count1, index=['middle_sum'], columns=['conc_flag'],
#                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
conc_by_role2.reset_index(inplace=True)
conc_by_role2.columns = ['inner_sum','no_conc','conc']
conc_by_role2['total'] = conc_by_role2['no_conc'] + conc_by_role2['conc']
conc_by_role2['rate'] = conc_by_role2['conc'] / conc_by_role2['total']

chi2, p, dof, expected = scs.chi2_contingency(conc_by_role2[['no_conc','conc']])
print('p-value, concussion vs non-concussions by # of midfield defenders')
print(p)

#
#
#
# test if interior left or right side overload (0, 1, 2, etc) is related to concussion rate
#
#
#

conc_by_role3 = pd.pivot_table(role_count1, index=['interior_side_overload_grp'], columns=['conc_flag'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
#run this if you want all possible values
#conc_by_role3 = pd.pivot_table(role_count1, index=['interior_side_overload'], columns=['conc_flag'],
#                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
conc_by_role3.reset_index(inplace=True)
conc_by_role3.columns = ['overload_sum','no_conc','conc']
conc_by_role3['total'] = conc_by_role3['no_conc'] + conc_by_role3['conc']
conc_by_role3['rate'] = conc_by_role3['conc'] / conc_by_role3['total']

chi2, p, dof, expected = scs.chi2_contingency(conc_by_role3[['no_conc','conc']])
print('p-value, concussion vs non-concussions by # of L or R overloaded defenders')
print(p)

#
#
# test if number of offensive interior players  is related to concussion rate. 
# Value possibilities ore 0 through 4.
#
#

conc_by_role4 = pd.pivot_table(role_count1, index=['offense_interior_grp'], columns=['conc_flag'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
#run this if you want all possible values
#conc_by_role4 = pd.pivot_table(role_count1, index=['offense_interior'], columns=['conc_flag'],
#                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
conc_by_role4.reset_index(inplace=True)
conc_by_role4.columns = ['offense_interior_sum','no_conc','conc']
conc_by_role4['total'] = conc_by_role4['no_conc'] + conc_by_role4['conc']
conc_by_role4['rate'] = conc_by_role4['conc'] / conc_by_role4['total']

chi2, p, dof, expected = scs.chi2_contingency(conc_by_role4[['no_conc','conc']])
print('p-value, concussion vs non-concussions by # of interior offensive players')
print(p)

#plot rates from concussions by role groupings
y_pos = np.arange(len(conc_by_role1['outer_sum']))
plt.ylabel('Concussion Rate (per 1000 plays)')
plt.xlabel('Number of jammers')
plt.bar(y_pos, conc_by_role1['rate']*1000, tick_label=conc_by_role1['outer_sum'], 
        color=['#1f77b4','r','crimson'])

# number of jammers is strongly associated with concussion rates, confirming my initial hunch
# however, the results are opposite what I had initially expected. I thought double-teamed gunners
# would've resulted in fewer concussions, as the gunners were, what I thought anecdotally,the 
# players most likely to crash into someone at high speed.
# Now we look at the potential impact a rule change would have on other aspects of the game 
# around punt plays.

# The above analysis shows that the number of concussions are statistically significantly different
# based on the number of jammers. The difference in concussion rates based on the number of 
# L or R side overload by the interior defenders was almost significant, so for funsies we'll 
# include that here too

# After submitting, I realized I probably should've written a function here for this. I did this
# analysis iteratively, first only looking at the outer sum, and would've gained nothing at that 
# time writing a function. Lesson learned!
#
#
# jammers and touchdown rates
#
#

print('See how number of jammers ')
outer_sum_td_rate = pd.pivot_table(role_count1, index=['outer_sum_grp'], columns=['touchdown'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
outer_sum_td_rate.reset_index(inplace=True)
outer_sum_td_rate.columns = ['outer_sum_grp','no_td','td']
outer_sum_td_rate['total'] = outer_sum_td_rate['no_td'] + outer_sum_td_rate['td']
outer_sum_td_rate['rate'] = outer_sum_td_rate['td'] / outer_sum_td_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(outer_sum_td_rate[['no_td','td']])
print('p-value, # of jammers and touchdown rates')
print(p)

#
#
# jammers and touchback rates
#
#

outer_sum_tb_rate = pd.pivot_table(role_count1, index=['outer_sum_grp'], columns=['touchback'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
outer_sum_tb_rate.reset_index(inplace=True)
outer_sum_tb_rate.columns = ['outer_sum_grp','no_tb','tb']
outer_sum_tb_rate['total'] = outer_sum_tb_rate['no_tb'] + outer_sum_tb_rate['tb']
outer_sum_tb_rate['rate'] = outer_sum_tb_rate['tb'] / outer_sum_tb_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(outer_sum_tb_rate[['no_tb','tb']])
print('p-value, # of jammers and touchback rates')
print(p)
#touchback rates are higher when there are only 2 jammers
#may be related to teams not attempting returns when punts are within reach of a touchback
#analysis of average/median punt team yard line would probably show 2-gunner formations to be 
#across the 50 more often
#
#
# jammers and punt block rates
#
#

outer_sum_pb_rate = pd.pivot_table(role_count1, index=['outer_sum_grp'], columns=['blocked'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
outer_sum_pb_rate.reset_index(inplace=True)
outer_sum_pb_rate.columns = ['outer_sum_grp','no_pb','pb']
outer_sum_pb_rate['total'] = outer_sum_pb_rate['no_pb'] + outer_sum_pb_rate['pb']
outer_sum_pb_rate['rate'] = outer_sum_pb_rate['pb'] / outer_sum_pb_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(outer_sum_pb_rate[['no_pb','pb']])
print('p-value, # of jammers and blocked punt rates')
print(p)
#punt block rates are lower when there are 4 jammers, which follows common sense 
#(fewer rushing, intention of punt defenders is to maximize return yards)


#
#
# jammers and penalty rates (regardless of on who, type, or whether accepted or declined)
#
#

outer_sum_plty_rate = pd.pivot_table(role_count1, index=['outer_sum_grp'], columns=['penalty'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
outer_sum_plty_rate.reset_index(inplace=True)
outer_sum_plty_rate.columns = ['outer_sum_grp','no_plty','plty']
outer_sum_plty_rate['total'] = outer_sum_plty_rate['no_plty'] + outer_sum_plty_rate['plty']
outer_sum_plty_rate['rate'] = outer_sum_plty_rate['plty'] / outer_sum_plty_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(outer_sum_plty_rate[['no_plty','plty']])
print('p-value, # of jammers and penalty rates')
print(p)
#penalties are significantly less frequent in 0-2 jammer formation

#
#
# jammers and fair catch rates 
#
#

outer_sum_fc_rate = pd.pivot_table(role_count1, index=['outer_sum_grp'], columns=['fair_catch'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
outer_sum_fc_rate.reset_index(inplace=True)
outer_sum_fc_rate.columns = ['outer_sum_grp','no_fc','fc']
outer_sum_fc_rate['total'] = outer_sum_fc_rate['no_fc'] + outer_sum_fc_rate['fc']
outer_sum_fc_rate['rate'] = outer_sum_fc_rate['fc'] / outer_sum_fc_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(outer_sum_fc_rate[['no_fc','fc']])
print('p-value, # of jammers and penalty rates')
print(p)
#fair catches are significantly more frequent in 0-2 jammer formation

#
#
# Interior side overload defenders and touchdown rates
#
#

interior_side_overload_td_rate = pd.pivot_table(role_count1, index=['interior_side_overload_grp'], columns=['touchdown'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
interior_side_overload_td_rate.reset_index(inplace=True)
interior_side_overload_td_rate.columns = ['inner_sum','no_conc','conc']
interior_side_overload_td_rate['total'] = interior_side_overload_td_rate['no_conc'] + interior_side_overload_td_rate['conc']
interior_side_overload_td_rate['rate'] = interior_side_overload_td_rate['conc'] / interior_side_overload_td_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(interior_side_overload_td_rate[['no_conc','conc']])
print('p-value, # of Interior side overload defenders and touchdown rates')
print(p)

#
#
# Interior side overload defenders and touchback rates
#
#

interior_side_overload_tb_rate = pd.pivot_table(role_count1, index=['interior_side_overload_grp'], columns=['touchback'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
interior_side_overload_tb_rate.reset_index(inplace=True)
interior_side_overload_tb_rate.columns = ['interior_side_overload_grp','no_conc','conc']
interior_side_overload_tb_rate['total'] = interior_side_overload_tb_rate['no_conc'] + interior_side_overload_tb_rate['conc']
interior_side_overload_tb_rate['rate'] = interior_side_overload_tb_rate['conc'] / interior_side_overload_tb_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(interior_side_overload_tb_rate[['no_conc','conc']])
print('p-value, # of Interior side overload defenders and touchback rates')
print(p)
#
#
# Interior side overload defenders and punt block rates
#
#

interior_side_overload_pb_rate = pd.pivot_table(role_count1, index=['interior_side_overload_grp'], columns=['blocked'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
interior_side_overload_pb_rate.reset_index(inplace=True)
interior_side_overload_pb_rate.columns = ['interior_side_overload_grp','no_conc','conc']
interior_side_overload_pb_rate['total'] = interior_side_overload_pb_rate['no_conc'] + interior_side_overload_pb_rate['conc']
interior_side_overload_pb_rate['rate'] = interior_side_overload_pb_rate['conc'] / interior_side_overload_pb_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(interior_side_overload_pb_rate[['no_conc','conc']])
print('p-value, # of Interior side overload defenders and blocked punt rates')
print(p)

#
#
# Interior side overload defenders and penalty rates (regardless of on who, type, or whether accepted or declined)
#
#

interior_side_overload_plty_rate = pd.pivot_table(role_count1, index=['interior_side_overload_grp'], columns=['penalty'],
                              aggfunc=lambda x: len(x))['PlayID'].fillna(0)
interior_side_overload_plty_rate.reset_index(inplace=True)
interior_side_overload_plty_rate.columns = ['interior_side_overload_grp','no_conc','conc']
interior_side_overload_plty_rate['total'] = interior_side_overload_plty_rate['no_conc'] + interior_side_overload_plty_rate['conc']
interior_side_overload_plty_rate['rate'] = interior_side_overload_plty_rate['conc'] / interior_side_overload_plty_rate['total']

chi2, p, dof, expected = scs.chi2_contingency(interior_side_overload_plty_rate[['no_conc','conc']])
print('p-value, # of Interior side overload defenders and penalty rates')
print(p)
#none of the results from interior side overload were statistically significant.

#print contingency tables / crosstabs for visual analysis and inclusion in presentation
print('\n Rate of touchdowns by number of jammers')
print(outer_sum_td_rate)
print('\n Rate of touchback by number of jammers')
print(outer_sum_tb_rate)
print('\n Rate of penalties by number of jammers')
print(outer_sum_plty_rate)
print('\n Rate of fair catches by number of jammers')
print(outer_sum_fc_rate)

#get global velocity data by role combined with concussion event velocities by role, to see if there's 
#insights in this part of the dataset, related to the formation

#mean velocities for the concussed player
temp1 = concussion_events.groupby(['Role'])['max_mph_player','avg_mph_player'].agg({'mean'}).reset_index()
temp1.columns=['Role','max_mph_player_avg','avg_mph_player_avg']

#mean velocities for the partner player, when applicable
temp2 = concussion_events.groupby(['partner_role'])['max_mph_partner','avg_mph_partner'].agg({'mean'}).reset_index()
temp2.columns=['Role','max_mph_partner_avg','avg_mph_partner_avg']

temp3 = pd.merge(temp1, temp2, how='outer', on='Role')

#add counts of concussions by player + partner role
temp4 = concussion_events.groupby(['Role'])['conc_flag'].agg({'sum'}).reset_index()
temp4.columns=['Role','conc_count_player']

temp5 = concussion_events.groupby(['partner_role'])['conc_flag'].agg({'sum'}).reset_index()
temp5.columns=['Role','conc_count_partner']

temp6 = pd.merge(temp4, temp5, how='outer', on='Role')
temp7 = pd.merge(temp3, temp6, how='outer', on='Role')

#merge with velocity data for all punt plays
conc_speed_compare = pd.merge(punt_role_ngs, temp7, how='outer', on='Role')

#calculate differences. Using overall max speed, which would represent the worst possible collision velocity 
#(even if the concussions didn't happen at the max recorded velocity)
conc_speed_compare['max_speed_diff_player'] = conc_speed_compare['max_mph_avg'] - conc_speed_compare['max_mph_player_avg']
conc_speed_compare['max_speed_diff_partner'] = conc_speed_compare['max_mph_avg'] - conc_speed_compare['max_mph_partner_avg']

print('statistics on the deviation of concussed player velocity compared to global average maximum speed, in MPH')
print('max', conc_speed_compare['max_speed_diff_player'].max())
print('median', conc_speed_compare['max_speed_diff_player'].median())
print('mean', conc_speed_compare['max_speed_diff_player'].mean())

print('statistics on the deviation of partner player velocity compared to global average maximum speed, in MPH')
print('max', conc_speed_compare['max_speed_diff_partner'].max())
print('median', conc_speed_compare['max_speed_diff_partner'].median())
print('mean', conc_speed_compare['max_speed_diff_partner'].mean())

#this result shows that the concussed players weren't going as fast as players in general
#further in-depth analysis of player velocities, while interesting from a data science perspective,
#are of limited probative value, since the formation is shown to have such a strong association
#with concussion rates, and formations are readily easy to modify via rules. The player behaviors
#would follow naturally as a result.

#plot rates from touchdowns by role groupings
y_pos = np.arange(len(outer_sum_td_rate['outer_sum_grp']))
plt.ylabel('Touchdowns per 1000 plays')
plt.xlabel('Number of jammers')
plt.bar(y_pos, outer_sum_td_rate['rate']*1000, tick_label=outer_sum_td_rate['outer_sum_grp'], 
        color=['g','#1f77b4','#1f77b4'])

#plot rates from touchbacks by role groupings
y_pos = np.arange(len(outer_sum_td_rate['outer_sum_grp']))
plt.ylabel('Touchbacks per 1000 plays')
plt.xlabel('Number of jammers')
plt.bar(y_pos, outer_sum_tb_rate['rate']*1000, tick_label=outer_sum_tb_rate['outer_sum_grp'], 
        color=['g','#1f77b4','#1f77b4'])

#plot rates from penalties by role groupings
y_pos = np.arange(len(outer_sum_td_rate['outer_sum_grp']))
plt.ylabel('Penalties per 1000 plays')
plt.xlabel('Number of jammers')
plt.bar(y_pos, outer_sum_plty_rate['rate']*1000, tick_label=outer_sum_plty_rate['outer_sum_grp'], 
        color=['g','#1f77b4','#1f77b4'])

