import numpy as np
import pandas as pd 
import json

fpl2020_file = open('../input/fantasy-epl-new-season-research-2020-2021/FPL_2019_20_season_stats.jscsrc')
fpl2020 = fpl2020_file.read()
fpl2020 = json.loads(fpl2020)
fpl2021_file = open('../input/fantasy-epl-new-season-research-2020-2021/FPL_2020_21_player_list.jscsrc')
fpl2021 = fpl2021_file.read()
fpl2021 = json.loads(fpl2021)
fpl2020.keys()
for key in fpl2020.keys():
    print('Data type: %s for Key: %s' %(type(fpl2020[key]),key))
print('Understanding the data structure for Teams')
print(fpl2020['teams'][0].keys())
print('Understanding the data structure for Elements')
print(fpl2020['elements'][0].keys())
teams2020 = pd.DataFrame(fpl2020['teams'])
players2020 = pd.DataFrame(fpl2020['elements'])
teams2020.head()
players2020.head()
fpoints_table = players2020.groupby('team_code')['total_points'].sum()
fpoints_table = pd.DataFrame(fpoints_table)
fpoints_table['code'] = fpoints_table.index
fpoints_table['fpoints_rank']=fpoints_table['total_points'].rank(ascending = False)
leaguetable_2020 = teams2020[['short_name','code','win','draw','loss','points','position','strength','strength_overall_home','strength_overall_away','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away','pulse_id']]
league_fpoints = leaguetable_2020.join(fpoints_table, on='code', how='left',lsuffix = 'lt')
table_comparison = league_fpoints[['short_name','position','points','fpoints_rank','total_points','code']]
table_comparison.sort_values('position')
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = table_comparison['points'].values.reshape(-1,1)
y = table_comparison['total_points'].values.reshape(-1,1)
names = table_comparison['short_name'].values.reshape(-1,1)
model.fit(X,y)
y_pred = model.predict(X)
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(X,y,color='black')
plt.plot(X,y_pred, color='blue', linewidth=3)

for a,b,l in zip(X,y,names):
    plt.annotate(l,(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.show()

fpoints_by_pos = players2020.groupby(by = ['team_code','element_type'],as_index=False)['total_points'].sum()
fpoints_pos = fpoints_by_pos.pivot(index='team_code', columns='element_type', values='total_points')
fpoints_pos.columns = ['GK','Def','Mid','Fwd']
fpoints_pos['code']=fpoints_pos.index
fp_pos = table_comparison.join(fpoints_pos, on='code', how='left',lsuffix = 'lt')
for each in ['GK','Def','Mid','Fwd']:
    fp_pos['perc_'+str(each)]=(fp_pos[each]/fp_pos['total_points']*100).round(2)
fp_pos['perc_Defense'] = fp_pos['perc_GK'] + fp_pos['perc_Def']
fp_pos[['short_name', 'position','total_points', 'perc_GK', 'perc_Def',
       'perc_Mid', 'perc_Fwd','code','perc_Defense']].sort_values('total_points',ascending = False)
players2021 = pd.DataFrame(fpl2021['elements'])
combined = players2020.merge(players2021[[ 'now_cost','code','element_type','team_code']], on = 'code',how='outer',suffixes = ['','_21'])
teams2020['team_code'] = teams2020['code'].astype('Int64')
combined['team_code'] = combined['team_code'].astype('Int64')
combined = combined.merge(teams2020, on = 'team_code', how = 'left',suffixes = ('','_team'))
def points_calc(row,label='element_type'):
    points = 0
    if row[label] == 1:
        points = points + 6*row['goals_scored'] + 4*row['clean_sheets'] + 5*row['penalties_saved'] # + save_points - 2goals concededpts
    elif row[label] == 2:
        points = points + 6*row['goals_scored'] + 4*row['clean_sheets'] #  - 2goals concededpts
    elif row[label] == 3:
        points = points + 5*row['goals_scored'] + 1*row['clean_sheets']
    else:
        points = points + 4*row['goals_scored']
    
    points = points + 3*row['assists'] - 1*row['yellow_cards'] - 2*row['red_cards'] - 2*row['own_goals'] -2*row['penalties_missed'] + row['bonus']
    return points
players2020['projected_points'] = players2020.apply(points_calc,axis = 1)
players2020[['projected_points','total_points']]
players2020['gw_points'] = players2020['total_points'] - players2020['projected_points']
combined = players2020.merge(players2021[[ 'now_cost','code','element_type','team_code']], on = 'code',how='outer',suffixes = ['','_21'])
combined['projected_points_21'] = combined.apply(points_calc,axis = 1,args = ['element_type_21'] )
combined['proj_tot_points_21'] = combined['projected_points_21'] + combined['gw_points']
combined['team_code'] = combined['team_code'].astype('Int64')
combined['team_code_21'] = combined['team_code_21'].astype('Int64')
combined = combined.merge(teams2020, on = 'team_code', how = 'left',suffixes = ('','_team'))
combined = combined.merge(teams2020, left_on = 'team_code_21',right_on='team_code', how = 'left',suffixes = ('','_team_21'))
fpoints_table_21 = combined.groupby('team_code')['proj_tot_points_21'].sum()
fpoints_table_21 = pd.DataFrame(fpoints_table_21)
fpoints_table_21['code'] = fpoints_table.index
fpoints_table_21['fpoints_rank']=fpoints_table_21['proj_tot_points_21'].rank(ascending = False)
leaguetable_2021 = teams2020[['short_name','code','win','draw','loss','points','position','strength','strength_overall_home','strength_overall_away','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away','pulse_id']]
league_fpoints_21 = leaguetable_2021.join(fpoints_table_21, on='code', how='left',lsuffix = 'lt')
table_comparison_21 = league_fpoints_21[['short_name','position','points','fpoints_rank','proj_tot_points_21','code']]
table_comparison_21 = table_comparison_21.merge(table_comparison[['code','total_points']], on='code', how='left',suffixes = ['','_20'])
table_comparison_21 = table_comparison_21.sort_values('position')
table_comparison_21[:-3]
fpoints_by_pos_21 = combined.groupby(by = ['team_code','element_type_21'],as_index=False)['proj_tot_points_21'].sum()
fpoints_pos_21 = fpoints_by_pos_21.pivot(index='team_code', columns='element_type_21', values='proj_tot_points_21')
fpoints_pos_21.columns = ['GK','Def','Mid','Fwd']
fpoints_pos_21['code']=fpoints_pos_21.index
fp_pos_21 = table_comparison_21.join(fpoints_pos_21, on='code', how='left',lsuffix = 'lt')
for each in ['GK','Def','Mid','Fwd']:
    fp_pos_21['proj_perc_'+str(each)]=(fp_pos_21[each]/fp_pos_21['proj_tot_points_21']*100).round(2)
fp_pos_21['proj_perc_Defense'] = fp_pos_21['proj_perc_GK'] + fp_pos_21['proj_perc_Def']
fp_pos_21 = fp_pos_21[['short_name', 'position','proj_tot_points_21', 'proj_perc_GK', 'proj_perc_Def',
       'proj_perc_Mid', 'proj_perc_Fwd','code','proj_perc_Defense']].sort_values('proj_tot_points_21',ascending = False)
fp_pos_21[:-3] 
# At a player level there is some need for basic filtering the list of players.
#Remove players that do not have a team code for 2021. 
targets = combined.dropna(axis=0,how = 'any',subset = ['team_code_21'])
# Remove players that dont have a Projected points score or the score is 0
targets = targets.dropna(axis=0,how = 'any',subset = ['proj_tot_points_21'])
targets = targets[targets['proj_tot_points_21']!=0]
#Remove players that have 0 minutes played
targets = targets[targets['minutes']!=0]
#Remove players that have lesser than 2 projected points per game
targets['proj_ppg_21'] = targets['proj_tot_points_21']/round(targets['total_points'] / targets['points_per_game'].astype(float) , 0)
targets = targets [targets['proj_ppg_21']>=2]
targets.shape[0]
players_per_club = targets['team_code_21'].value_counts()
targets.pivot_table(values ='team_code_21',index =  'short_name_team_21',aggfunc = 'count')
targets['proj_points_per90'] = round(targets['proj_tot_points_21'] / targets['minutes']*90,2)
targets['proj_points_permn_per90'] = round(targets['proj_tot_points_21'] / (targets['now_cost_21']/10)/ targets['minutes']*90,3)
club_proj_scores = targets.groupby(by = 'team_code_21')['proj_tot_points_21'].sum()
targets['club_points_21'] =  targets['team_code_21'].apply(lambda x: club_proj_scores[x])
targets['proj_club_points_percent'] = round(targets['proj_tot_points_21'] / targets['club_points_21']*100,2)
targets['proj_bonus_points_perc'] = round(targets['bonus'] / targets['proj_tot_points_21']*100,2)

targets[['proj_points_per90','proj_points_permn_per90','proj_club_points_percent','proj_bonus_points_perc']].corr()
# Plotting Points per 90 vs Total Points for the first 30 players by total points
data = targets[['proj_points_per90','proj_tot_points_21','web_name']].nlargest(50,'proj_tot_points_21')
plt.figure(figsize=(15,10))
plt.scatter(data['proj_tot_points_21'],data['proj_points_per90'],color='black')
for a,b,l in zip(data['proj_tot_points_21'],data['proj_points_per90'],data['web_name']):
    plt.annotate(l,(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per 90')

plt.show()

data = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','code_team_21']].nlargest(50,'proj_tot_points_21')
plt.figure(figsize=(15,10))
plt.scatter(data['proj_tot_points_21'],data['proj_points_permn_per90'],c=data['code_team_21'],cmap = 'tab20')
for a,b,l in zip(data['proj_tot_points_21'],data['proj_points_permn_per90'],data['web_name']):
    plt.annotate(l,(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per million per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per million per 90')
plt.show()
gkdata = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','element_type_21','now_cost_21','code_team_21']]

gkdata = gkdata[gkdata['element_type_21']==1].nlargest(25,'proj_tot_points_21')
plt.figure(figsize=(15,10))
plt.scatter(gkdata['proj_tot_points_21'],gkdata['proj_points_permn_per90'],c=gkdata['code_team_21'],cmap = 'tab20')
for a,b,l,c in zip(gkdata['proj_tot_points_21'],gkdata['proj_points_permn_per90'],gkdata['web_name'],gkdata['now_cost_21']):
    plt.annotate(l+ ' '+ str(c/10),(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per million per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per million per 90')
plt.show()
defchrt = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','element_type_21','now_cost_21','code_team_21']]

defchrt = defchrt[defchrt['element_type_21']==2].nlargest(50,'proj_tot_points_21')
plt.figure(figsize=(15,10))
plt.scatter(defchrt['proj_tot_points_21'],defchrt['proj_points_permn_per90'],c=defchrt['code_team_21'],cmap = 'tab20')
for a,b,l,c in zip(defchrt['proj_tot_points_21'],defchrt['proj_points_permn_per90'],defchrt['web_name'],defchrt['now_cost_21']):
    plt.annotate(l+ ' '+ str(c/10),(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per million per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per million per 90')
plt.show()
defchrt = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','element_type_21','now_cost_21','code_team_21']]

defchrt = defchrt[defchrt['element_type_21']==2].nlargest(50,'proj_tot_points_21')[4:]
defchrt = defchrt[defchrt['proj_points_permn_per90']<1]
plt.figure(figsize=(15,10))
plt.scatter(defchrt['proj_tot_points_21'],defchrt['proj_points_permn_per90'],c=defchrt['code_team_21'],cmap = 'tab20')
for a,b,l,c in zip(defchrt['proj_tot_points_21'],defchrt['proj_points_permn_per90'],defchrt['web_name'],defchrt['now_cost_21']):
    plt.annotate(l+ ' '+ str(c/10),(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per million per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per million per 90')
plt.show()
midchrt = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','element_type_21','now_cost_21','code_team_21']]

midchrt = midchrt[midchrt['element_type_21']==3].nlargest(50,'proj_tot_points_21')
plt.figure(figsize=(15,10))
plt.scatter(midchrt['proj_tot_points_21'],midchrt['proj_points_permn_per90'],c=midchrt['code_team_21'],cmap = 'tab20')
for a,b,l,c in zip(midchrt['proj_tot_points_21'],midchrt['proj_points_permn_per90'],midchrt['web_name'],midchrt['now_cost_21']):
    plt.annotate(l+ ' '+ str(c/10),(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per million per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per million per 90')
plt.show()
midchrt = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','element_type_21','now_cost_21','code_team_21']]

midchrt = midchrt[midchrt['element_type_21']==3].nlargest(50,'proj_tot_points_21')[9:]

midchrt = midchrt[midchrt['proj_points_permn_per90']<1]
plt.figure(figsize=(15,10))
plt.scatter(midchrt['proj_tot_points_21'],midchrt['proj_points_permn_per90'],c=midchrt['code_team_21'],cmap = 'tab20')
for a,b,l,c in zip(midchrt['proj_tot_points_21'],midchrt['proj_points_permn_per90'],midchrt['web_name'],midchrt['now_cost_21']):
    plt.annotate(l+ ' '+ str(c/10),(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.title('Points per million per 90 vs Total Points')
plt.xlabel('Total Points')
plt.ylabel('Points per million per 90')
plt.show()


fwdchrt = targets[['proj_points_permn_per90','proj_tot_points_21','web_name','element_type_21','now_cost_21','code_team_21']]

fwdchrt = fwdchrt[fwdchrt['element_type_21']==4].nlargest(35,'proj_tot_points_21')
plt.figure(figsize=(15,10))
plt.scatter(fwdchrt['proj_tot_points_21'],fwdchrt['proj_points_permn_per90'],c=fwdchrt['code_team_21'],cmap = 'tab20')
for a,b,l,c in zip(fwdchrt['proj_tot_points_21'],fwdchrt['proj_points_permn_per90'],fwdchrt['web_name'],fwdchrt['now_cost_21']):
    plt.annotate(l+ ' '+ str(c/10),(a,b), textcoords="offset points", xytext=(0,7), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.show()