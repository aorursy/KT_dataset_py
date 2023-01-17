import numpy as np 

import pandas as pd 

import seaborn as sns

import datetime

import json

import plotly.express as px

import plotly.graph_objects as go



matches = pd.read_csv('../input/epl-2019-2020-match-data/match_infos_EPL_1920.csv',index_col = 'Unnamed: 0')

rosters = pd.read_csv('../input/epl-2019-2020-match-data/rosters_EPL_1920.csv',index_col = 'Unnamed: 0')

plays = pd.read_csv('../input/epl-2019-2020-match-data/shots_EPL_1920.csv',index_col = 'Unnamed: 0')

fpl_player_history = pd.read_csv('../input/fantasy-epl-new-season-research-2020-2021/player_history_2020_21.csv')

fpl2020_file = open('../input/fantasy-epl-new-season-research-2020-2021/FPL_2019_20_season_stats.jscsrc')

fpl2020 = fpl2020_file.read()

fpl2020 = json.loads(fpl2020)



fpl_gws = pd.DataFrame(fpl2020['events'])

fpl_teams = pd.DataFrame(fpl2020['teams'])

fpl_players = pd.DataFrame(fpl2020['elements'])

fpl_teams = fpl_teams.rename(columns={'code': 'team_code'})



#formatting dates

for df in [matches,plays]:

    df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)

fpl_gws['deadline_time'] = pd.to_datetime(fpl_gws['deadline_time'],infer_datetime_format = True)



#Removing gwks where the league was suspeded due to covid

fpl_gws = pd.concat([fpl_gws, fpl_gws['top_element_info'].apply(lambda x: pd.Series({'top_element_info_id':x['id'],'top_element_info_points':x['points']}))], axis = 1)

fpl_gws = fpl_gws[fpl_gws['top_element_info_points']!=0].copy()

fpl_gws.index = list(range(1,39))

fpl_gws.reset_index().drop(['index'],axis = 1)





# Creating a fantasy gw column in epl matches by finding the closest deadline date which is lesser than match date. 

matches['fpl_gw_deadline'] = matches['date'].apply(lambda x: (x + max([date - x for date in fpl_gws['deadline_time'] if x > date])))



# Outputs the fantasy gw as the index value of the gameweek deadline which is less than the match date, and closest to the match date. 

matches['fpl_gw_id'] = matches['fpl_gw_deadline'].apply(lambda x: fpl_gws['deadline_time'][fpl_gws['deadline_time'] == x].index.values[0])



#Create teams table from epl data

epl_teams_1920 = matches[['a','team_a']].drop_duplicates(subset=['a'], keep='first', inplace=False, ignore_index=True)

epl_teams_1920.columns = ['team_id','epl_team_name']



#Create players table from rosters data

epl_players_1920 = rosters[['player_id','player','team_id']].drop_duplicates(subset=['player_id'], keep='last', inplace=False, ignore_index=True)



#Create teams table from fpl data and match them on fullnames of teams

team_fullnames = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton & Hove Albion', 'Burnley',

       'Chelsea', 'Crystal Palace', 'Everton', 'Leicester', 'Liverpool',

       'Manchester City', 'Manchester United', 'Newcastle United', 'Norwich', 'Sheffield United',

       'Southampton', 'Tottenham Hotspurs', 'Watford', 'West Ham United', 'Wolverhampton Wanderers']



fullnames_df = pd.DataFrame(data = [fpl_teams.name.values,team_fullnames])

fullnames_df = fullnames_df.transpose()

fullnames_df.columns = ['name','full_club_name']

fpl_teams = fpl_teams.merge(fullnames_df,on = ['name'])





#Create a table that matches team names using fuzzy wuzzy

from fuzzywuzzy import fuzz

from fuzzywuzzy import process

epl_teams_1920['matched_full_name'] = epl_teams_1920['epl_team_name'].apply(lambda x: process.extractOne(str(x),team_fullnames)[0])

fpl_teams['matched_full_name'] = fpl_teams['name'].apply(lambda x: process.extractOne(str(x),team_fullnames)[0])

# Get fpl team id for matched table

matched_teams = epl_teams_1920.merge(fpl_teams[['name','matched_full_name','team_code']],on = ['matched_full_name'])

epl_players_1920_teamcodes_merged = epl_players_1920.merge(matched_teams,how = 'left', on = ['team_id'])



# We are going to match the player_id key in the epl players table and the code key in the fpl table using the player names. 

# To do that we will match the name of a player in the fpl table all the players in the same club in the fpl table. This allows us to reduce our error rate and processing time. 

# A downside is that the players who switched clubs in the EPL itself will face a problem. We will make a list of such players to rework the matching.





# Use the fpl team_code in matched_epl players table to get the list of all names of players at that club and use fuzzywuzzy to match with epl player name

fpl_players['fullname'] = fpl_players['first_name']+' '+ fpl_players['second_name']

epl_players_1920_teamcodes_merged[['fpl_name','match_confidence']] = epl_players_1920_teamcodes_merged.apply(lambda row: pd.Series(process.extractOne(str(row['player']), fpl_players[fpl_players['team_code']==row['team_code']]['fullname'].tolist())),axis = 1)

# We are going to retain the confidence level of the match unlike the earlier club names matching. This is beacause matching is a lot lot harder with about 500 player names that with 20 club names. 



# We are going to rework on the players that did not get 100% match confidence

matched_correctly = epl_players_1920_teamcodes_merged[epl_players_1920_teamcodes_merged['match_confidence']>=90]

rework_name_matching = epl_players_1920_teamcodes_merged[epl_players_1920_teamcodes_merged['match_confidence']<90]

matched_correctly_merged = matched_correctly.merge(fpl_players[['fullname','code']], how = 'left',left_on = 'fpl_name',right_on='fullname')



#Filter the fpl dataset for unmatched players

remaining_fpl_players = fpl_players[~fpl_players.code.isin(matched_correctly_merged.code.unique().tolist())]



#Rerun the matching using the unmatched databases only, this time we use web_name from the fpl table, instead of full_name

remaining_fpl_players = fpl_players[~fpl_players.code.isin(matched_correctly_merged.code.unique().tolist())]

rework_name_matching = rework_name_matching.drop(['fpl_name','match_confidence'],axis = 1)

rework_name_matching[['fpl_name','match_confidence']] = rework_name_matching.apply(lambda row: pd.Series(process.extractOne(str(row['player']), remaining_fpl_players[remaining_fpl_players['team_code']==row['team_code']]['fullname'].tolist())),axis = 1)



# Another run at name matching to get the rest of the names 

matched_correctly_merged_1 = rework_name_matching[rework_name_matching['match_confidence']>85]

# join by web_name to get fpl code

matched_correctly_merged_1 = matched_correctly_merged_1.merge(fpl_players[['fullname','code']], how = 'left',left_on = 'fpl_name',right_on='fullname')





#Another round of matching



remaining_fpl_players_2 = remaining_fpl_players[~remaining_fpl_players['code'].isin(matched_correctly_merged_1.code.unique().tolist())]

rework_name_matching_2 = rework_name_matching[rework_name_matching['match_confidence']<=85]

rework_name_matching_2 = rework_name_matching_2.drop(['fpl_name','match_confidence'],axis = 1)

rework_name_matching_2[['fpl_name','match_confidence']] = rework_name_matching_2.apply(lambda row: pd.Series(process.extractOne(str(row['player']), remaining_fpl_players_2[remaining_fpl_players_2['team_code']==row['team_code']]['fullname'].tolist() , scorer=fuzz.token_sort_ratio)),axis = 1)



matched_correctly_merged_2 = rework_name_matching_2[rework_name_matching_2['match_confidence']>80]

matched_correctly_merged_2 = matched_correctly_merged_2.merge(fpl_players[['fullname','code']], how = 'left',left_on = 'fpl_name',right_on='fullname')



# If we look at the unmatched players, they are mostly because the epl table has their name as a short name. So we will do another round of matching using the web_name

# However this approach may result duplicates as limited information is used to make an approximate guess. 



remaining_fpl_players_3 = remaining_fpl_players[~remaining_fpl_players['code'].isin(matched_correctly_merged_2.code.unique().tolist())]

rework_name_matching_3 = rework_name_matching_2[rework_name_matching_2['match_confidence']<=85]

rework_name_matching_3 = rework_name_matching_3.drop(['fpl_name','match_confidence'],axis = 1)

rework_name_matching_3[['fpl_name','match_confidence']] = rework_name_matching_3.apply(lambda row: pd.Series(process.extractOne(str(row['player']), remaining_fpl_players_2[remaining_fpl_players_2['team_code']==row['team_code']]['web_name'].tolist() , scorer=fuzz.token_sort_ratio)),axis = 1)

matched_correctly_merged_3 = rework_name_matching_3[rework_name_matching_3['match_confidence']>85]

matched_correctly_merged_3 = matched_correctly_merged_3.merge(fpl_players[['web_name','code']], how = 'left',left_on = 'fpl_name',right_on='web_name')



# Replace web_name by the full name to preserve the commonality

matched_correctly_merged_3['fpl_name'] = matched_correctly_merged_3.apply(lambda row: fpl_players[fpl_players['code']==row['code']]['fullname'].values[0],axis = 1)

matched_correctly_merged_3.columns = matched_correctly_merged.columns



epl_fpl_players_matched = pd.concat([matched_correctly_merged,matched_correctly_merged_1,matched_correctly_merged_2,matched_correctly_merged_3],axis = 0)

remaining = epl_players_1920[~epl_players_1920['player_id'].isin(epl_fpl_players_matched['player_id'].tolist())]



# Just Cenk Tosun, and Solly March are not matched. We will just fix that manually in the interest of time.  



remaining_matched = remaining.drop(['team_id'],axis = 1)

remaining_matched['fullname'] = [fpl_players[fpl_players['web_name']=='Tosun']['fullname'].values[0],fpl_players[fpl_players['web_name']=='March']['fullname'].values[0]]

remaining_matched['code'] = [fpl_players[fpl_players['web_name']=='Tosun']['code'].values[0],fpl_players[fpl_players['web_name']=='March']['code'].values[0]]



epl_fpl_players_matched_2 = pd.concat([epl_fpl_players_matched,remaining_matched])



# Also a quick check for dups will reveal some duplicates and we will need to delete the duplicate entries. 

# We have just 1 duplicate entry

counts = epl_fpl_players_matched_2.groupby(by=['player_id']).count()

dups = counts[counts['player']>1].index.tolist()

dups_df = epl_fpl_players_matched_2[epl_fpl_players_matched_2['player_id'].isin(dups)]



# We need to delete the duplicate entry for Jota that has player_id as 2132 and cpde as 89274, Again this is a bit manual and not ideal

epl_fpl_players_matched_final = epl_fpl_players_matched_2[~((epl_fpl_players_matched_2['player_id']==2132) & (epl_fpl_players_matched_2['code']==89274))]





# As a final item, we are going to add a category for the team that season based on where they finished. And short club names from FPL

club_id_mapping = matched_teams[['team_id', 'matched_full_name', 'team_code']]

club_category = pd.Series(['relg_fight','top4','top_half','top_half','top_half','midtable','relg_fight','top6_cont','top6_cont','top4','top_half','relg_fight','midtable','relg_fight','midtable','top4','top6_cont','midtable','top6_cont','top4'],name = 'category')

club_id_mapping = pd.concat([club_id_mapping,club_category],axis = 1)

club_category_rankings = {'relg_fight':5, 'top4':1, 'top_half':3, 'midtable':4, 'top6_cont':2}

club_id_mapping['club_category_rank'] = club_id_mapping['category'].apply(lambda x: club_category_rankings[x] )

club_id_mapping = club_id_mapping.merge(fpl_teams[['short_name','team_code','position']], how = 'left', on = 'team_code')

# After all that heavy lifting we finally have two mapping tables for OPTA teams to FPL teams and OPTA players to FPL Players



player_id_mapping = epl_fpl_players_matched_final[['player_id','player','fullname','code']]
#assign points to every match

matches['winner'] = matches.apply(lambda row: 'h' if row['h_goals'] > row['a_goals'] else 'd' if row['h_goals'] == row['a_goals'] else 'a',axis = 1) 

matches['h_points'] = matches['winner'].apply(lambda x: 3 if x == 'h' else 1 if x == 'd' else 0) 

matches['a_points'] = matches['winner'].apply(lambda x: 0 if x == 'h' else 1 if x == 'd' else 3) 



#build season stats at club level 

seasonstats = pd.DataFrame(club_id_mapping['team_id'])

for stat in ['a_deep', 'a_goals', 'a_ppda', 'a_shot', 'a_shotOnTarget', 'a_xg', 'h_deep', 'h_goals', 'h_ppda','h_shot','h_shotOnTarget', 'h_xg', 'h_points', 'a_points']:

    if stat.split('_')[0]=='h':

        stat = matches.groupby(by=['h'])[stat].sum().reset_index()

        stat = stat.rename(columns = {'h':'team_id'})

    else:

        stat = matches.groupby(by=['a'])[stat].sum().reset_index()

        stat = stat.rename(columns = {'a':'team_id'})

    seasonstats = seasonstats.merge(stat, on = 'team_id')



seasonstats = seasonstats.merge(club_id_mapping[['team_id','short_name','team_code']], how = 'left', on = 'team_id')

seasonstats = seasonstats.merge(fpl_teams[['team_code','position']], how = 'left', on = 'team_code')

seasonstats = seasonstats.sort_values('position')

seasonstats['total_points'] = seasonstats['h_points']+seasonstats['a_points']



points_chart = seasonstats[['short_name','h_points','a_points']]

points_chart.columns = ['Club','Home Points','Away Points']

fig = px.bar(points_chart, x='Club', y=['Home Points', 'Away Points'], title='Total Points - Home vs Away')

fig.update_layout(

    yaxis_title="Points",

    legend_title="Turf",

)

fig.show()
#Create table of home points and away points by club category of the opposition

# Merge home and away categories for home and away teams and relabel the columns

matches = matches.merge(club_id_mapping[['team_id','category','club_category_rank']],how = 'left', left_on = 'h', right_on = 'team_id')

matches = matches.rename(columns = {'team_id':'h_team_id','category':'h_club_category','club_category_rank':'h_club_category_rank'})

matches = matches.merge(club_id_mapping[['team_id','category','club_category_rank']],how = 'left', left_on = 'a', right_on = 'team_id')

matches = matches.rename(columns = {'team_id':'a_team_id','category':'a_club_category','club_category_rank':'a_club_category_rank'})



#Group the matches for home teams by opposition category

home_opp_category_points = matches.groupby(by = ['h','a_club_category_rank'],as_index = False)['h_points'].agg({'points_vs_oppcat': 'sum', 'games_vs_opp_cat': 'count'})

home_opp_category_points['turf'] = 'h'

home_opp_category_points = home_opp_category_points.rename(columns = {'h':'team_id','a_club_category_rank':'opp_club_category_rank'})

#Group the matches for away teams by opposition category

away_opp_category_points = matches.groupby(by = ['a','h_club_category_rank'],as_index = False)['a_points'].agg({'points_vs_oppcat': 'sum', 'games_vs_opp_cat': 'count'})

away_opp_category_points['turf'] = 'a'

away_opp_category_points = away_opp_category_points.rename(columns = {'a':'team_id','h_club_category_rank':'opp_club_category_rank'})



# Combine home and away and create points per game

points_vs_opp_cat = pd.concat([home_opp_category_points,away_opp_category_points])

points_vs_opp_cat['points_per_game'] = round(points_vs_opp_cat['points_vs_oppcat']/points_vs_opp_cat['games_vs_opp_cat'],2)

points_vs_opp_cat = points_vs_opp_cat.merge(club_id_mapping[['team_id','category','team_code']], how = 'left', on = 'team_id')

points_vs_opp_cat = points_vs_opp_cat.merge(fpl_teams[['position','team_code','short_name']], how = 'left', on = 'team_code')

points_vs_opp_cat = points_vs_opp_cat.sort_values(['position','opp_club_category_rank'])

points_vs_opp_cat['opp_category_name'] = points_vs_opp_cat['opp_club_category_rank'].apply(lambda x: club_id_mapping[club_id_mapping['club_category_rank']==x]['category'].values[0])



# Group by turf to get 

points_vs_opp_cat_combined = points_vs_opp_cat.groupby(['team_id','opp_category_name'], as_index = False)['points_per_game'].mean()

points_vs_opp_cat_combined = points_vs_opp_cat_combined.merge(points_vs_opp_cat[['team_id','position', 'short_name']], how = 'left', on = 'team_id')



points_vs_opp_cat_combined['opp_club_category_rank'] = points_vs_opp_cat_combined['opp_category_name'].apply(lambda x : club_id_mapping[club_id_mapping['category']==x]['club_category_rank'].values[0])

points_vs_opp_cat_combined = points_vs_opp_cat_combined.sort_values(['position','opp_club_category_rank'])



fig = px.scatter(points_vs_opp_cat_combined, x='short_name', y='opp_category_name', color='opp_category_name',size='points_per_game', hover_data=['points_per_game'])

fig.update_layout(

    title = 'Points per game split by Opposition Strength',

    xaxis_title='Club',

    yaxis_title='Opposition Strength',

    legend_title='Opposition Strength',

)

fig.show()
# points earned vs team category home vs away 



fig = px.scatter(points_vs_opp_cat, x='short_name', y='opp_category_name', color='opp_category_name',size='points_per_game', hover_data=['points_per_game'], facet_col = 'turf')

fig.update_layout(

    title = 'Points per game split by Opposition Strength',

    yaxis_title='Opposition Strength',

    legend_title='Opposition Strength',

)

fig.for_each_annotation(lambda a: a.update(text = ('Home' if a.text.split("=")[-1]=='h' else 'Away')))



fig.show()
team_match_stats = pd.DataFrame()

for team_id in matches['a'].unique():        

    team_matches_home = matches[matches['h_team_id']==team_id]

    team_matches_home = team_matches_home.rename(lambda x: 'opp_' + x.split('_')[-1] if all([x.split('_')[0]=='a', len(x)>2]) else x, axis=1)

    team_matches_home = team_matches_home.rename(lambda x: 'self_' + x.split('_')[-1] if all([x.split('_')[0]=='h', len(x)>2]) else x, axis=1)

    team_matches_home['turf'] = 'h'

    team_matches_away = matches[matches['a_team_id']==team_id]

    team_matches_away = team_matches_away.rename(lambda x: 'opp_' + x.split('_')[-1] if x.split('_')[0]=='h' else x, axis=1)

    team_matches_away = team_matches_away.rename(lambda x: 'self_' + x.split('_')[-1] if x.split('_')[0]=='a' else x, axis=1)

    team_matches_away['turf'] = 'a'

    team_stats = pd.concat([team_matches_home, team_matches_away], axis = 0)

    team_stats['team_id'] = team_id

    team_stats = team_stats.dropna(axis = 1)

    team_match_stats = pd.concat([team_match_stats,team_stats],axis = 0)



team_match_stats = team_match_stats.merge(club_id_mapping[['team_id','category','club_category_rank','short_name','position']],how = 'left', on = 'team_id')



team_season_stats = team_match_stats.groupby('team_id', as_index = False).sum()

team_season_stats = team_season_stats.drop(['fid', 'id', 'fpl_gw_id','self_id', 'self_rank','opp_id', 'opp_rank', 'club_category_rank',],axis = 1)

for column in [ 'league_id', 'season', 'position']:

    team_season_stats[column] = team_season_stats[column].apply( lambda x: int(x/38))

    

team_season_stats = team_season_stats.merge(club_id_mapping[['team_id','category','club_category_rank','short_name']],how = 'left', on = 'team_id')



team_season_stats = team_season_stats.sort_values('position')

fig = px.bar(team_season_stats, x='short_name', y='opp_points',hover_data=['self_points'], title='Points Conceded')

fig.update_layout(

    xaxis_title="Club",

    yaxis_title="Points",

)

fig.show()
team_season_stats = team_season_stats.sort_values('position')

fig = px.bar(team_season_stats, x='short_name', y='self_ppda',hover_data=['opp_goals'], title='Park the Bus Score')

fig.update_layout(

    xaxis_title="Club",

    yaxis_title="Passes allowed per defensive action",

)

fig.show()
plays_goals = plays[plays['result']=='Goal']

plays_goals_players = plays_goals.groupby(['player_id','situation'],as_index = False)['result'].count()

plays_goals_players = plays_goals_players.merge(epl_fpl_players_matched_final[['team_id', 'player_id']],how = 'left', on = 'player_id')

plays_goals_teams = plays_goals_players.groupby(['team_id','situation'],as_index = False)['result'].sum()

plays_goals_teams = plays_goals_teams.merge(club_id_mapping[['team_id', 'short_name','position']],how = 'left', on = 'team_id')



plays_goals_teams_pvt = plays_goals_teams.pivot(index = 'team_id',columns = 'situation', values = 'result' ).reset_index()

plays_goals_teams_pvt = plays_goals_teams_pvt.merge(club_id_mapping[['team_id', 'short_name','position']],how = 'left', on = 'team_id')

plays_goals_teams_pvt = plays_goals_teams_pvt.sort_values(['position'])

fig = px.bar(plays_goals_teams_pvt, x='short_name', y=['DirectFreekick','FromCorner','OpenPlay','Penalty','SetPiece'], title='Goals scored from')

fig.update_layout(

    xaxis_title='Clubs',

    yaxis_title='Goals',

    legend_title='Situation Type',

)

fig.show()
player_season_stats = rosters.groupby(['team_id','player_id'],as_index = False)[['goals','assists','shots','key_passes','xG','xA','xGBuildup', 'xGChain']].sum()

player_season_stats = player_season_stats.merge(club_id_mapping[['team_id', 'category','club_category_rank', 'short_name']],how = 'left',on = 'team_id')

player_season_stats = player_season_stats.merge(player_id_mapping[['player_id', 'code']],how = 'left', on = 'player_id')

player_season_stats = player_season_stats.merge(fpl_players[['code','total_points', 'element_type','points_per_game','minutes','assists','clean_sheets','goals_conceded','bonus','web_name']],how = 'left', on = 'code',suffixes=['_epl','_fpl'])

player_season_stats = player_season_stats.sort_values(['team_id','total_points'],ascending = False)

player_season_stats_top5 = player_season_stats.groupby(['team_id','player_id'], as_index = False).apply(lambda df: df[0:5]).reset_index()

p_s_s_top5_scaled = player_season_stats_top5.copy()

attrs =  ['goals', 'assists_epl', 'shots', 'key_passes','xG', 'xA', 'xGBuildup', 'xGChain','assists_fpl']

p_s_s_top5_scaled[attrs] = p_s_s_top5_scaled[attrs] / p_s_s_top5_scaled[attrs].max() 

p_s_s_top5_scaled_stats = p_s_s_top5_scaled[['team_id','player_id', 'goals', 'assists_epl','shots', 'key_passes']]

p_s_s_top5_scaled_stats_melted = p_s_s_top5_scaled_stats.melt(id_vars = ['team_id','player_id'],value_vars = [ 'goals', 'assists_epl','shots', 'key_passes'])

p_s_s_top5_scaled_stats_melted = p_s_s_top5_scaled_stats_melted.merge(player_season_stats[['player_id','total_points','short_name','web_name']], how = 'left', on = 'player_id')

p_s_s_top5_scaled_stats_melted = p_s_s_top5_scaled_stats_melted.sort_values(['total_points','variable'], ascending = False)



p_s_s_top5_scaled = p_s_s_top5_scaled.sort_values('total_points',ascending = False)



radar_cols = ['goals', 'assists_fpl', 'shots', 'key_passes']

fig = go.Figure()

for i in range(2):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled[radar_cols].iloc[i],

          theta=radar_cols,

          fill='toself',

          name=p_s_s_top5_scaled['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = 'De Bruyne vs Salah')



fig.show()
fig = go.Figure()

for i in range(5):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled[radar_cols].iloc[i],

          theta=radar_cols,

          fill='toself',

          name=p_s_s_top5_scaled['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = "Top 5 FPL players")



fig.show()
p_s_s_top5_scaled_fwds = p_s_s_top5_scaled[p_s_s_top5_scaled['element_type'] == 4] 

fig = go.Figure()

for i in range(5):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled_fwds[radar_cols].iloc[i],

          theta=radar_cols,

          fill='toself',

          name=p_s_s_top5_scaled_fwds['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = "Top 5 Forwards")



fig.show()
p_s_s_top5_scaled_fwds = p_s_s_top5_scaled[p_s_s_top5_scaled['element_type'] == 3] 

fig = go.Figure()

for i in range(5):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled_fwds[radar_cols].iloc[i],

          theta=radar_cols,

          fill='toself',

          name=p_s_s_top5_scaled_fwds['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = "Top 5 Midfielders")



fig.show()


p_s_s_top5_scaled_fwds = p_s_s_top5_scaled[p_s_s_top5_scaled['element_type'] == 4] 



radar_cols_underlying = ['key_passes', 'xG', 'xA', 'xGBuildup', 'xGChain']

fig = go.Figure()

for i in range(5):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled_fwds[radar_cols_underlying].iloc[i],

          theta=radar_cols_underlying,

          fill='toself',

          name=p_s_s_top5_scaled_fwds['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = "Underlying Stats: Top 5 Forwards")



fig.show()


p_s_s_top5_scaled_mids = p_s_s_top5_scaled[(p_s_s_top5_scaled['element_type'] == 3)|(p_s_s_top5_scaled['code']==169187)] 



radar_cols_underlying = ['key_passes', 'xG', 'xA', 'xGBuildup', 'xGChain']

fig = go.Figure()

for i in range(6):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled_mids[radar_cols_underlying].iloc[i],

          theta=radar_cols_underlying,

          fill='toself',

          name=p_s_s_top5_scaled_mids['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = "Underlying Stats: Top 5 Midfielders")



fig.show()


p_s_s_top5_scaled_defs = p_s_s_top5_scaled[(p_s_s_top5_scaled['element_type'] == 2)] 



radar_cols_underlying = ['key_passes', 'xG', 'xA', 'xGBuildup', 'xGChain']

fig = go.Figure()

for i in range(5):

    fig.add_trace(go.Scatterpolar(

          r=p_s_s_top5_scaled_defs[radar_cols_underlying].iloc[i],

          theta=radar_cols_underlying,

          fill='toself',

          name=p_s_s_top5_scaled_defs['web_name'].iloc[i]))





fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True, title = "Underlying Stats: Top 5 Defenders")



fig.show()
plays['field_x'] = 7-plays['Y']*7

plays['field_y'] = plays['X']*10



def create_full_field(fig):

    # Set axes properties

    fig.update_xaxes(range=[-1, 8], showgrid=False)

    fig.update_yaxes(range=[-1, 11])



    # Add shapes

    fig.add_shape(type="rect",x0=-0.5, y0=-0.5, x1=7.5, y1=10.5,

                  line=dict(color="white",width=2),fillcolor="green", layer = 'below') # full pitch



    fig.add_shape(type="rect",x0=0, y0=0, x1=7, y1=10,

                  line=dict(color="white",width=5), layer = 'below')# full playing area



    fig.add_shape(type="rect",x0=0, y0=0, x1=7, y1=5,

                  line=dict(color="white",width=4),layer = 'below')# home half

    

    fig.add_shape(type="circle",fillcolor="green",x0=3.5-0.915,y0=1.1-0.915,x1=3.5+0.915,y1=1.1+0.915,

                  line=dict(color="white",width=2),layer = 'below')# home penalty arc

    

    fig.add_shape(type="rect",x0=1.5, y0=0, x1=5.5, y1=1.65,

                  line=dict(color="white",width=3),fillcolor="green",layer = 'below')# home penalty box

    

    fig.add_shape(type="rect",x0=1.5+1.65-.55, y0=0, x1=5.5-1.65+.55, y1=0.55,

                  line=dict(color="white",width=3),layer = 'below')# home inner penalty box

    

    fig.add_shape(type="rect",x0=1.5+1.65, y0=-0.15, x1=5.5-1.65, y1=0,

                  line=dict(color="grey",width=1),fillcolor="white",layer = 'below'

                 )# home goal

    fig.add_trace(go.Scatter(x=[3.5],y=[1.1],mode="markers",

                             marker=dict(color='white',size=5,),showlegend = False))#home penalty spot



    fig.add_shape(type="circle",fillcolor="green",x0=3.5-0.915,y0=10-(1.1-0.915),x1=3.5+0.915,y1=10-(1.1+0.915),

                line=dict(color="white",width=2),layer = 'below')# away penalty arc

    

    fig.add_shape(type="rect",x0=1.5, y0=10, x1=5.5, y1=10-1.65,

                line=dict(color="white",width=3),fillcolor="green",layer = 'below')# away penalty box



    fig.add_shape(type="rect",x0=1.5+1.65-.55, y0=10-0.55, x1=5.5-1.65+.55, y1=10,

                line=dict(color="white",width=3),layer = 'below')# away inner penalty box

    

    fig.add_shape(type="rect",x0=1.5+1.65, y0=10+0.15, x1=5.5-1.65, y1=10,

                line=dict(color="grey",width=1),fillcolor="white",layer = 'below')# away goal

    

    fig.add_trace(go.Scatter(x=[3.5],y=[10-1.1],mode="markers",

                             marker=dict(color='white',size=5),showlegend = False)) #away penalty spot



    fig.add_trace(go.Scatter( x=[3.5],y=[5],mode="markers",

                             marker=dict(color='white',size=10),showlegend = False))#centre spot



    fig.add_shape(type="circle",x0=3.5-0.915,y0=5-0.915,x1=3.5+0.915,y1=5+0.915,

                  line=dict(color="white",width=2),layer = 'below')# centre circle



    fig.update_xaxes(showticklabels=False,showgrid=False,zeroline=False,title='',)#hide x-axis

    fig.update_yaxes(showticklabels=False,showgrid=False,zeroline=False,title='',)#hide y-axis

    fig.update_layout(autosize=False,width=700,height=700,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)') #fix size

    return fig



salah_shots_field = px.scatter(plays[plays['player_id']==1250].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation','shotType','player_assisted'], title = 'Salah Shots Heat Map' )   



salah_shots_field = create_full_field(salah_shots_field)                 

salah_shots_field.show()

mane_shots_field = px.scatter(plays[plays['player_id']==838].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation'], title = 'Mane Shots Heat Map')      



mane_shots_field = create_full_field(mane_shots_field)                 

mane_shots_field.show()

kdb_shots_field = px.scatter(plays[plays['player_id']==750].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation'], title = 'Kevin DeBruyne Shots Heat Map')      



kdb_shots_field = create_full_field(kdb_shots_field)                 

kdb_shots_field.show()
auba_shots_field = px.scatter(plays[plays['player_id']==318].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation'], title = 'Aubamayeng Shots Heat Map')      



auba_shots_field = create_full_field(auba_shots_field)                 

auba_shots_field.show()
sterling_shots_field = px.scatter(plays[plays['player_id']==618].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation'], title = 'Sterling Shots Heat Map')      



sterling_shots_field = create_full_field(sterling_shots_field)                 

sterling_shots_field.show()
ings_shots_field = px.scatter(plays[plays['player_id']==986].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation'], title = 'Ings Shots Heat Map')      



ings_shots_field = create_full_field(ings_shots_field)                 

ings_shots_field.show() 
fernandes_shots_field = px.scatter(plays[plays['player_id']==1228].sort_values('result'), x="field_x", y="field_y", color="result",

                 size='xG', hover_data=['situation'], title = 'Bruno Fernandes Shots Heat Map')      



fernandes_shots_field = create_full_field(fernandes_shots_field)                 

fernandes_shots_field.show()
goals_sankey = plays.groupby(['situation','shotType'],as_index = False)['player_id'].count()

enconding = {'DirectFreekick':0,'FromCorner':1,'OpenPlay':2,'Penalty':3,'SetPiece':4,'LeftFoot':5,'RightFoot':6,'Head':7,'OtherBodyPart':8,'BlockedShot':9,'Goal':10,'MissedShots':11,'SavedShot':12,'ShotOnPost':13,'OwnGoal':14}

goals_sankey = goals_sankey.replace(enconding)

goals_sankey_result = plays.groupby(['shotType','result'],as_index = False)['player_id'].count()

goals_sankey_result = goals_sankey_result.replace(enconding)



fig = go.Figure(data=[go.Sankey(

    node = dict(

      pad = 15,

      thickness = 20,

      line = dict(color = "grey", width = 0.5),

      label = list(enconding),

      color = "green"

    ),

    link = dict(

      source = goals_sankey['situation'].append(goals_sankey_result['shotType']), # indices correspond to labels, eg A1, A2, A2, B1, ...

      target = goals_sankey['shotType'].append(goals_sankey_result['result']),

      value = goals_sankey['player_id'].append(goals_sankey_result['player_id'])

  ))])



fig.update_layout(title_text="All the shots from last season.", font_size=10)

fig.show()
reel = plays[plays['match_id']==11643]

reel = reel.sort_values('minute')

reel['field_x'] = 7-plays['Y']*7

reel['field_y'] = reel.apply(lambda row: row['X']*10 if row['h_a']=='h' else 10 - row['X']*10, axis = 1)

reel['play_text'] = reel.apply(lambda row: str(row['minute']) + ' minute: ' + row['player'] + ' ' + row['result'], axis = 1)



reel_trail =  px.scatter(reel, x='field_x', y='field_y', animation_frame="minute",text = 'play_text',

           size="xG", color="result", hover_name="player", size_max=55, range_x=[0,11], range_y=[0,11],

                        labels = 'result')



reel_trail = create_full_field(reel_trail)

reel_trail.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2500

reel_trail.update_traces(textposition='top center',textfont_color='black',textfont_size=15)

reel_trail.update_layout(showlegend = False)



reel_trail.show()