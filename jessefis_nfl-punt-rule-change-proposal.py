import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import datetime
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import re
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
injuries = pd.read_csv('../input/video_review.csv', delimiter=',')
plays = pd.read_csv('../input/play_information.csv', delimiter=',')
positions = pd.read_csv('../input/player_punt_data.csv', delimiter=',')
roles = pd.read_csv('../input/play_player_role_data.csv', delimiter=',')
# group positions into groups

punt_return = {
'role_return_line':['PDL1','PDL2','PDL3','PDL4','PDL5','PDL6','PDM','PDR1','PDR2','PDR3','PDR4','PDR5','PDR6'],
'role_return_lead_blocker':['PFB'],
'role_return_linebacker':['PLL','PLL1','PLL2','PLL3','PLR','PLR1','PLR2','PLR3','PLM','PLM1'],
'role_return_returner':['PR'],
'role_return_jammer':['VR','VR','VL','VLi','VLo','VRi','VRo']}

punt_team = {
'role_punt_line':['PLG','PLS','PLT','PRG','PRT'],
'role_punt_wings':['PLW','PRW'],
'role_punt_punter':['P'],
'role_punt_protector':['PPR','PC','PPL','PPLi','PPLo','PPRi','PPRo'],
'role_punt_gunner':['GL','GLi','GLo','GR','GRi','GRo']}

roles['punt_or_return'] = np.nan
roles['role_group'] = np.nan
for k,v in punt_return.items():
    roles[k] = np.where(roles['Role'].isin(v), 1, 0)
    roles['punt_or_return'] = np.where(roles['Role'].isin(v), 'return', roles['punt_or_return'])
    roles['role_group'] = np.where(roles['Role'].isin(v), k, roles['role_group'])
for k,v in punt_team.items():
    roles[k] = np.where(roles['Role'].isin(v), 1, 0)
    roles['punt_or_return'] = np.where(roles['Role'].isin(v), 'punt', roles['punt_or_return'])
    roles['role_group'] = np.where(roles['Role'].isin(v), k, roles['role_group'])
roles.head()
# create a categorical for each role group based on whether there were more/less players relative a typical punt

roles_per_play = roles[['GameKey','PlayID','punt_or_return','role_group',
    'role_return_line','role_return_lead_blocker','role_return_linebacker','role_return_returner','role_return_jammer',
    'role_punt_line','role_punt_punter','role_punt_protector','role_punt_gunner','role_punt_wings']].groupby(['GameKey','PlayID']).sum()

for group in list(punt_return)+list(punt_team):
    median = int(roles_per_play[group].median())
    # hack for now since median is 2 but most common 2,3,4 - maybe should have used pd.cut
    if group == 'role_return_jammer':
        median = 3
    print(group, median, roles_per_play[group].value_counts().to_dict())
    roles_per_play[group] = pd.cut(roles_per_play[group], [-1, median-0.5, median + 0.5, 100], labels=[str(median-1)+'_or_less',str(median),str(median+1)+'_or_more'])
    print(roles_per_play[group].value_counts().to_dict())
roles_per_play.head()
# The punt team doesn't seem to have much variation, however the return team has some flexability which might be useful to take advantage of!
# one row per play
df = plays.copy()

# merge with concussions
df = df.merge(injuries[['GameKey','PlayID','Player_Activity_Derived','Turnover_Related','Primary_Impact_Type',
                             'GSISID','Friendly_Fire']],how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])

# remove plays which dont actually end up with a punt 
df = df[~df['PlayDescription'].str.contains('BLOCKED|Aborted|False Start| pass |Delay of Game')]
# note that this removes a concussion play sample: "J.Ryan up the middle to LA 47 for 26 yards. FUMBLES, recovered by SEA-N.Thorpe at LA 40. SEA-J.Ryan was injured during the play.  Los Angeles challenged the loose ball recovery ruling, and the play was Upheld. The ruling on the field stands. (Timeout #2.)']"
df = df[df['PlayDescription'].str.contains('punts')]
print(len(df), len(plays))

# merge to get role (for the injured player)
df = df.merge(roles[['GameKey','PlayID','GSISID','Role','punt_or_return','role_group']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])

# merge to get number of each role on the play
df = df.merge(roles_per_play,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])

# merge with positions (for injured player) - note that positions has dups
df = df.merge(positions[['GSISID','Position']].drop_duplicates(subset=['GSISID']),how='left',left_on=['GSISID'],right_on=['GSISID'])

# hack - removing a bunch of text that made my life more difficult to parse
df['cleaned_description'] = df['PlayDescription'].str.replace('FUMBLES.*|recovers.*|RECOVERED.*|recovered.*|Officially.*|.*REVERSED|challenged the kick downed|downed the ball at the .*, but was ruled out-of-bounds|for the remainder|for a concussion','')

# whether or not there was a concussion
df['had_concussion'] = np.where(df['Primary_Impact_Type'].isnull(), 0, 1)

# tagging the event types
event_types = ['downed','Touchback','fair catch','for -*[0-9]* yard','for no gain','MUFFS','out of bounds\.']
for text in event_types:
    df[text] = np.where(df['cleaned_description'].str.contains(text), 1, 0)
    print(text, df[text].sum(), df[df[text]>0]['had_concussion'].sum())
    
# make sure all rows only have one tagged event type
print('dropping invalid tagged events=', df[df[event_types].sum(axis=1)!=1]['PlayDescription'].values)
df = df[df[event_types].sum(axis=1)==1]

print(len(df), len(plays), df.columns)
# punts yards
df['punt_yards'] = df['PlayDescription'].str.replace('.* punts ','').str.replace(' yard.*','').astype(int)

# return yards
df['return_yards'] = np.where(df['for -*[0-9]* yard']==1, 
                                  df['cleaned_description'].str.replace('.* for| yard.*|-yds.*',''), 
                                  np.where(df['for no gain']==1, 0, np.nan)).astype(float)

# which side of field punting from
df['punt_from_own'] = df.apply(lambda row: row['Poss_Team'] in row['YardLine'], axis=1)

# yard being punted from (0->100 where 0 is the punting teams own end zone)
df['punt_from'] = df['YardLine'].str.replace('.* ','').astype(int)
df['punt_from'] = np.where(df['punt_from_own'], df['punt_from'], 100-df['punt_from'])
df['line_of_scrimmage'] = df['punt_from']

# yard being punted to (0->100 where 0 is the return teams own end zone)
df['punt_to'] = 100 - (df['punt_from'] + df['punt_yards'].astype(float))

# whether or not a return was attempted
df['no_return_attempted'] = np.where((df['downed']==1)|(df['Touchback']==1)|(df['fair catch']==1)|(df['out of bounds\.']==1), 1, 0)

# convert to categoricals
df['punt_yards'] = pd.cut(df['punt_yards'], [-1,40,46,52,100], labels=['40_or_less','40_to_46','46_to_52','52_or_more'])
df['punt_from'] = pd.cut(df['punt_from'], [-1,10,20,40,100], labels=['within_10','10_to_25','25_to_40','40_or_more'])
df['punt_to'] = pd.cut(df['punt_to'], [-1,10,20,40,100], labels=['within_10','10_to_25','25_to_40','40_or_more'])
df['return_yards'] = pd.cut(df['return_yards'],[-100,-0.5,5.5,15,100], labels=['negative','0_to_5','5_to_15','more_than_15'])
df['return_yards'] = np.where(df['no_return_attempted']>0,'no_return_attempted',df['return_yards'])
df['return_yards'] = np.where(df['MUFFS']>0,'muffed_punt',df['return_yards'])
print(df.groupby(['had_concussion'])['had_concussion'].count().to_frame())
print(df.groupby(['punt_or_return','Player_Activity_Derived'])['Player_Activity_Derived'].count().to_frame())
print(df.groupby(['punt_or_return','Player_Activity_Derived'])['Player_Activity_Derived'].count().to_frame())
print(df.groupby(['punt_or_return','role_group'])['role_group'].count().to_frame())
X = df[['punt_yards','punt_from','punt_to','return_yards','role_return_lead_blocker', 'role_return_linebacker','role_return_line', 'role_return_jammer']]

for col in X.columns:
    print(col, X[col].value_counts().to_dict())

X = pd.get_dummies(X)
X = X.loc[:, (X != 0).any(axis=0)]
y = df['had_concussion']
print(X.columns)
model = LogisticRegression()
model = model.fit(X, y)

coeff = pd.DataFrame(model.coef_.T, X.columns, columns=['coeff']).sort_values('coeff',ascending=False)
print(coeff)
for prefix in ['role_return_lead_blocker','role_return_line_','role_return_lineb','role_return_jammer','punt_to','punt_from','return_yards','punt_yards']:
    print(coeff[coeff.index.str.startswith(prefix)])
# predict large return
X = df[['punt_yards','punt_from','punt_to','role_return_lead_blocker', 'role_return_linebacker','role_return_line', 'role_return_jammer']]
X = pd.get_dummies(X)
X = X.loc[:, (X != 0).any(axis=0)]
y = np.where(df['return_yards']=='more_than_15',1,0)

model = LogisticRegression()
model = model.fit(X, y)

coeff = pd.DataFrame(model.coef_.T, X.columns, columns=['coeff']).sort_values('coeff',ascending=False)
print(coeff[coeff.index.str.contains('role')])
# predict no return attempted
X = df[['punt_yards','punt_from','punt_to','role_return_lead_blocker', 'role_return_linebacker','role_return_line', 'role_return_jammer']]
X = pd.get_dummies(X)
X = X.loc[:, (X != 0).any(axis=0)]
y = np.where(df['return_yards']=='no_return_attempted',1,0)

model = LogisticRegression()
model = model.fit(X, y)

coeff = pd.DataFrame(model.coef_.T, X.columns, columns=['coeff']).sort_values('coeff',ascending=False)
print(coeff[coeff.index.str.contains('role')])
# predict muffed punt
X = df[['punt_yards','punt_from','punt_to','role_return_lead_blocker', 'role_return_linebacker','role_return_line', 'role_return_jammer']]
X = pd.get_dummies(X)
X = X.loc[:, (X != 0).any(axis=0)]
y = np.where(df['return_yards']=='muffed_punt',1,0)

model = LogisticRegression()
model = model.fit(X, y)

coeff = pd.DataFrame(model.coef_.T, X.columns, columns=['coeff']).sort_values('coeff',ascending=False)
print(coeff[coeff.index.str.contains('role')])
# load next gen stats to calculate some different metrics
next_gen = pd.DataFrame()
for f in ['NGS-2016-pre','NGS-2016-post','NGS-2016-reg-wk1-6','NGS-2016-reg-wk7-12','NGS-2016-reg-wk13-17',
    'NGS-2017-pre','NGS-2017-post','NGS-2017-reg-wk1-6','NGS-2017-reg-wk7-12','NGS-2017-reg-wk13-17']:
    cur = pd.read_csv('../input/%s.csv'%(f), delimiter=',')
    
    # time when the alignment should be set
    starting_event = cur[cur.Event.isin(['line_set','ball_snap'])].sort_values('Time').drop_duplicates(subset=['GameKey','PlayID'])[['GameKey','PlayID','Time']].rename(columns={'Time':'snap_start_time'})
    
    # throw out pre-snap data
    cur = cur.merge(starting_event,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])
    cur = cur[cur['Time'] >= cur['snap_start_time']]
    
    cur = cur.merge(df[['GameKey','PlayID','line_of_scrimmage']],how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID']).dropna(subset=['line_of_scrimmage'])
    
    # use the earliest datapoint for each play/player to get starting player location
    initial_location = cur.sort_values('Time').drop_duplicates(subset=['GameKey','PlayID','GSISID'])[['GameKey','PlayID','GSISID','x','y']].rename(columns={'x':'x_initial','y':'y_initial'})
    cur = cur.merge(initial_location,how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])
    
    # add punt timestamp
    punt_time = cur[cur['Event']=='punt'].sort_values('Time').drop_duplicates(subset=['GameKey','PlayID'])[['GameKey','PlayID','Time']].rename(columns={'Time':'punt_timestamp'})
    
    # now calculate when player crosses line of scrimmage
    cross_los = cur[['GameKey','PlayID','GSISID','line_of_scrimmage','x_initial','Time']].drop_duplicates()
    # if player started on left side of line of scrimmage then see when x > los+1, otherwise see when x < los-1
    cross_los = cross_los[((cur['x_initial']-10 < cur['line_of_scrimmage'])&(cur['x']-10 > cur['line_of_scrimmage']+1))|
                          ((cur['x_initial']-10 > cur['line_of_scrimmage'])&(cur['x']-10 < cur['line_of_scrimmage']-1))]
    cross_los = cross_los.sort_values('Time').drop_duplicates(subset=['GameKey','PlayID','GSISID'])[['GameKey','PlayID','GSISID','Time','line_of_scrimmage']].rename(columns={'Time':'time_crossed_los'})
    
    # lets ignore outliers like this:
    #  75.639999  51.799999  2016-08-12 01:45:02.800  0.00
    #  75.639999  51.799999  2016-08-12 01:45:02.900  0.00
    #  71.739998  53.310001  2016-08-12 01:45:03.000  4.18
    #  75.750000  51.740002  2016-08-12 01:44:58.500  0.00
    #  75.750000  51.740002  2016-08-12 01:44:58.600  0.00
    cur = cur[cur['dis'] < 1.4667] # this translates to roughly 30 mph
    
    max_distance = cur.groupby(['GameKey','PlayID','GSISID'])['dis'].max().to_frame().reset_index().rename(columns={'dis':'max_distance'})
    
    # calculate max_mph: 1 yard/second = 2.04545 miles/hour, then multiply by 10 since data points are tenth of a second
    max_distance['max_mph'] = max_distance['max_distance'] * 2.04545 * 10
    cur_next_gen = max_distance[['GameKey','PlayID','GSISID','max_mph']]
    
    # calculate total distance traveled on play
    total_distance = cur.groupby(['GameKey','PlayID','GSISID'])['dis'].sum().to_frame().reset_index().rename(columns={'dis':'total_distance'})
    cur_next_gen = cur_next_gen.merge(total_distance,how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])

    # merge other metrics
    cur_next_gen = cur_next_gen.merge(initial_location,how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])
    cur_next_gen = cur_next_gen.merge(cross_los,how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])
    cur_next_gen = cur_next_gen.merge(punt_time,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])
    
    next_gen = pd.concat([next_gen, cur_next_gen])

# http://static.nfl.com/static/content/public/image/rulebook/pdfs/12_Rule9_Scrimmage_Kick.pdf
# During a kick from scrimmage, only the end men (eligible receivers) on the line of scrimmage at
# the time of the snap, or an eligible receiver who is aligned or in motion behind the line and is more than
# one yard outside the end man, are permitted to advance more than one yard beyond the line before the
# ball is kicked. 
next_gen['crossed_line_before_punt'] = np.where(next_gen['punt_timestamp']>next_gen['time_crossed_los'],1,0)
next_gen['outside_numbers'] = np.where((next_gen['y_initial']<12)|(next_gen['y_initial']>(53.3-12)), 1, 0)
next_gen['within_5_of_los'] = np.where(np.abs((next_gen['x_initial']-10)-next_gen['line_of_scrimmage'])<=5, 1, 0)
next_gen['within_10_of_los'] = np.where(np.abs((next_gen['x_initial']-10)-next_gen['line_of_scrimmage'])<=10, 1, 0)
next_gen['within_15_of_los'] = np.where(np.abs((next_gen['x_initial']-10)-next_gen['line_of_scrimmage'])<=15, 1, 0)

next_gen['max_mph'].hist()
plt.show()
next_gen['total_distance'].hist()
plt.show()
next_gen['x_initial'].hist()
plt.show()
next_gen['y_initial'].hist()
plt.show()
next_gen.head()
print(next_gen.groupby(['GameKey','PlayID'])['x_initial'].count().value_counts().head(10))
print(next_gen[~((next_gen['x_initial']<0)|(next_gen['x_initial']>53.3))].groupby(['GameKey','PlayID'])['x_initial'].count().value_counts().head(10))
# Ideally I'd feed this data into the model but I dont trust it enough.
filtered = next_gen[~((next_gen['x_initial']<0)|(next_gen['x_initial']>53.3))]
filtered = filtered.merge(roles[['GameKey','PlayID','GSISID','punt_or_return']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])

# limit to punt return team
filtered = filtered[filtered['punt_or_return']=='return']
# limit to plays with exactly 11
have_11 = filtered.groupby(['GameKey','PlayID'])['punt_or_return'].count().to_frame().reset_index()
#have_11 = have_11[have_11['punt_or_return']==11][['GameKey','PlayID']]
filtered = have_11.merge(filtered,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])

filtered = filtered.merge(df[['GameKey','PlayID','had_concussion']],how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])
next_gen_play_stats = filtered[['outside_numbers','within_5_of_los','within_10_of_los','within_15_of_los','GameKey','PlayID','had_concussion']].groupby(['GameKey','PlayID','had_concussion']).sum()#agg({'count','sum'})
next_gen_play_stats.groupby('had_concussion').agg({'mean','count'})#['outside_numbers'].count()#.mean()#gg({'mean','count'})#merge(roles[['GameKey','PlayID','GSISID','role_group','punt_or_return']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])
#look at crossed before punt

filtered = next_gen[~((next_gen['x_initial']<0)|(next_gen['x_initial']>53.3))]
filtered = filtered.merge(roles[['GameKey','PlayID','GSISID','punt_or_return']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])

# limit to punt return team
filtered = filtered[filtered['punt_or_return']=='punt']
# limit to plays with exactly 11
have_11 = filtered.groupby(['GameKey','PlayID'])['punt_or_return'].count().to_frame().reset_index()
have_11 = have_11[have_11['punt_or_return']==11][['GameKey','PlayID']]
filtered = have_11.merge(filtered,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])

filtered = filtered.merge(df[['GameKey','PlayID','had_concussion']],how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])
next_gen_play_stats = filtered[['crossed_line_before_punt','outside_numbers','within_5_of_los','within_10_of_los','within_15_of_los','GameKey','PlayID','had_concussion']].groupby(['GameKey','PlayID','had_concussion']).sum()#agg({'count','sum'})
next_gen_play_stats.groupby('had_concussion').agg({'mean','count'})#['outside_numbers'].count()#.mean()#gg({'mean','count'})#merge(roles[['GameKey','PlayID','GSISID','role_group','punt_or_return']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])
concussion_players_speed = injuries.merge(next_gen[['GameKey','PlayID','GSISID','max_mph']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])
concussion_players_speed['Primary_Partner_GSISID'] = concussion_players_speed['Primary_Partner_GSISID'].replace('Unclear',-1).fillna(-1).astype(int)
concussion_players_speed['max_mph_partner'] = concussion_players_speed[['GameKey','PlayID','Primary_Partner_GSISID']].merge(next_gen[['GameKey','PlayID','GSISID','max_mph']],how='left',
    left_on=['GameKey','PlayID','Primary_Partner_GSISID'],right_on=['GameKey','PlayID','GSISID'])['max_mph'].values

plt.show()
ax = plt.axes()
sns.scatterplot(x="max_mph_partner", y="max_mph", data=concussion_players_speed, ax=ax)
ax.set_title('Max speed (MPH) of players involved in a concussion')
plt.show()
next_gen_with_roles = next_gen.merge(roles[['GameKey','PlayID','GSISID','role_group','punt_or_return']],how='left',left_on=['GameKey','PlayID','GSISID'],right_on=['GameKey','PlayID','GSISID'])

# for each play/role_group - get the max_mph/avg_distance across the players within that role_group
max_mph_per_play_role = next_gen_with_roles.groupby(['GameKey','PlayID','role_group','punt_or_return'])['max_mph'].max().to_frame().reset_index()
avg_distance_per_play_role = next_gen_with_roles.groupby(['GameKey','PlayID','role_group','punt_or_return'])['total_distance'].mean().to_frame().reset_index()

# now merge with roles_per_play data
max_mph_per_play_role = roles_per_play.merge(max_mph_per_play_role,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])
avg_distance_per_play_role = roles_per_play.merge(avg_distance_per_play_role,how='left',left_on=['GameKey','PlayID'],right_on=['GameKey','PlayID'])
# show impact on punt team max_mph by changing a return team position group
for role_group in ['role_return_line','role_return_linebacker','role_return_lead_blocker','role_return_jammer']:
    filtered = max_mph_per_play_role[(max_mph_per_play_role['punt_or_return']=='punt')&(max_mph_per_play_role['role_group']!='role_punt_punter')]
    pivot = filtered.groupby(['role_group',role_group])['max_mph'].mean().to_frame().reset_index().pivot(index='role_group',columns=role_group,values='max_mph')
    ax = plt.axes()
    sns.heatmap(pivot, annot=True, fmt='.1f', ax = ax, cmap='Greens')
    ax.set_title('Impact of %s on Max MPH'%(role_group))
    plt.show()
# show impact on punt team max_distance by changing a return team position group
for role_group in ['role_return_line','role_return_linebacker','role_return_lead_blocker','role_return_jammer']:
    filtered = avg_distance_per_play_role[(avg_distance_per_play_role['punt_or_return']=='punt')&(avg_distance_per_play_role['role_group']!='role_punt_punter')]
    pivot = filtered.groupby(['role_group',role_group])['total_distance'].mean().to_frame().reset_index().pivot(index='role_group',columns=role_group,values='total_distance')
    ax = plt.axes()
    sns.heatmap(pivot, annot=True, fmt='.1f', ax = ax, cmap='Blues')
    ax.set_title('Impact of %s on Avg Distance'%(role_group))
    plt.show()