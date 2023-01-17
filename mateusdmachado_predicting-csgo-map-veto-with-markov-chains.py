import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from os import listdir

pd.set_option('display.max_columns',100)

listdir('../input/csgo-professional-matches/')
base_dir = '../input/csgo-professional-matches/'

veto_df = pd.read_csv(base_dir+'picks.csv',low_memory=False)
# Reversing the rows in the DataFrame so that the DataFrame is sorted in ascending order by date.

veto_df = veto_df.iloc[::-1]
veto_df.index = range(1,veto_df.shape[0]+1)
veto_df.head()
maps = ['Cache','Cobblestone','Dust2','Inferno','Mirage','Nuke','Overpass','Train','Vertigo']
maps_and_zero = maps + ['0.0']
picks = ['t1_removed_1','t1_removed_2','t1_removed_3','t2_removed_1','t2_removed_2',
                't2_removed_3','t1_picked_1','t2_picked_1','left_over']
veto_df = veto_df[veto_df.loc[:,picks].isin(maps_and_zero).all(axis=1)]
gb = veto_df.groupby('system').system.count()
possible_systems = list(gb[gb>100].index)
possible_systems
start_date = '2012-01-01'
final_date = '2022-01-01'

roster_changes = {'NRG':['Evil Geniuses',     '2019-09-26',final_date],
                'Renegades':['100 Thieves',   '2019-11-07',final_date],
                 'SMASH':['GODSENT',          '2016-04-07',final_date],
                 'SK':['MIBR',                '2018-07-03',final_date],
                 'Grayhound':['Renegades',    '2019-11-19',final_date]}

for old_team, [new_team, start_date, final_date] in roster_changes.items():
    veto_df.loc[(veto_df.team_1==old_team) & (veto_df.date < start_date),['team_1']] = new_team
    veto_df.loc[(veto_df.team_2==old_team) & (veto_df.date < start_date),['team_2']] = new_team
first_team_order = {}
second_team_order = {}

first_team_order ['1212125'] = [5,np.NaN, 5,np.NaN,4,3,2]
second_team_order['1212125'] = [np.NaN,5, np.NaN,5,3,4,2]

first_team_order ['1234125'] = [5,np.NaN, 1,4,4,3,2]
second_team_order['1234125'] = [np.NaN,5, 4,1,3,4,2]

first_team_order ['1122215'] = [5,5,np.NaN,np.NaN,np.NaN,   4,2]
second_team_order['1122215'] = [np.NaN,np.NaN,5,4,4,   np.NaN,2]

first_team_order ['1234215'] = [5,np.NaN, 1,4,3,4,2]
second_team_order['1234215'] = [np.NaN, 5,4,1,4,3,2]

first_team_order ['1221125'] = [5,np.NaN,np.NaN, 4,4,3,2]
second_team_order['1221125'] = [np.NaN,5,5,      np.NaN,np.NaN,4,2]

first_team_order ['1212345'] = [5,np.NaN,5,np.NaN, 1,3,2]
second_team_order['1212345'] = [np.NaN,5,np.NaN,5, 3,1,2]
# actually removing other veto systems

def filter_odd_systems(veto_df, possible_systems):
    veto_df = veto_df[veto_df.system.isin(possible_systems)].copy()
    veto_df.loc[:,['system']] = veto_df.system.astype(str)+'5'

    gb  = veto_df.groupby('match_id').date.count()
    veto_df = veto_df[veto_df.match_id.isin(gb[gb==1].index)].copy()
    
    return veto_df
# ignoring teams with few matches present in the dataset

def get_relevant_teams(veto_df, min_matches):
    relevant_teams = pd.concat((veto_df.team_1,veto_df.team_2))
    
    gb = relevant_teams.groupby(relevant_teams).count()
    relevant_teams = gb[gb>min_matches].index
    veto_df = veto_df[(veto_df.team_1.isin(relevant_teams)) & (veto_df.team_2.isin(relevant_teams))]
    
    return veto_df, relevant_teams
#this function is only useful when there already is a trained model, as to not have to retrain everything. Not useful in this notebook

def get_new_teams(veto_df,teams_dict,relevant_teams):
    new_teams = []
    for x in relevant_teams:
        if x not in teams_dict.keys():
            new_teams.append(x)
    relevant_teams = relevant_teams.drop(new_teams)
    veto_df = veto_df[~((veto_df.team_1.isin(new_teams)) | (veto_df.team_2.isin(new_teams)))]
    
    return veto_df,relevant_teams, new_teams
def filter_data(veto_df, min_matches, possible_systems, include_new_teams):
    veto_df = filter_odd_systems(veto_df,possible_systems)
    veto_df,relevant_teams = get_relevant_teams(veto_df,min_matches)
    
    if include_new_teams == 0:
        veto_df,relevant_teams, new_teams = get_new_teams(veto_df,teams_dict,relevant_teams)
    else:
        new_teams = []
    
    return veto_df, relevant_teams, new_teams
teams_dict_columns = ['date','start_veto_%','opponent','match_id','best_of','system','Dust2','Inferno',
 'Mirage','Nuke','Overpass','Train','Vertigo','Cache','Cobblestone','diff_1','diff_else']
filter_data_params = {'veto_df' : veto_df,
                     'min_matches':15,
                     'possible_systems': possible_systems,
                      #### this option is only useful when there already is a trained model
                     'include_new_teams':1}
veto_df, relevant_teams, new_teams = filter_data(**filter_data_params)

veto_df.loc[:,maps] = np.NaN
veto_df.head()
def create_teams_dict(veto_df, start_elo, start_diff, span, relevant_teams):
    teams_dict = {}
    teams_curr_row = {}

    veto_df_columns = ['date','team_1','match_id','best_of','system']
    veto_df_columns.extend(maps)
    teams_columns = ['date','team_1','opponent','match_id','best_of','system','Dust2','Inferno','Mirage','Nuke','Overpass','Train','Vertigo','Cache','Cobblestone','diff_1','diff_else']

    first_row = pd.DataFrame([['2012-08-21',0.5,0,0,0,0,start_elo,start_elo,start_elo,start_elo,start_elo,start_elo,start_elo,start_elo,start_elo,start_diff,start_diff]],columns=teams_columns)
    for team in relevant_teams:
        teams_dict[team] = veto_df.loc[(veto_df.team_1 == team) | (veto_df.team_2 == team),veto_df_columns]
        teams_dict[team].loc[teams_dict[team]['team_1'] != team,'team_1'] = 0
        teams_dict[team].loc[teams_dict[team]['team_1'].astype(str) == team,'team_1'] = 1
        teams_dict[team].team_1 = teams_dict[team].team_1.ewm(span=span,min_periods=1).mean()

        opponents = pd.concat((veto_df[veto_df.team_1 == team].team_2,veto_df[veto_df.team_2 == team].team_1)).sort_index()
        teams_dict[team].insert(2,'opponent',opponents)

        teams_dict[team]['diff_1'] = 0
        teams_dict[team]['diff_else'] = 0

        teams_dict[team] = pd.concat((first_row, teams_dict[team]))

        teams_curr_row[team] = 0
    
    return teams_dict, teams_curr_row
#this function is only useful when there already is a trained model, as to not have to retrain everything. Not useful in this notebook

def get_last_index():
    for index in veto_df.iloc[::-1].index:
        try:
            loc = teams_dict[veto_df.loc[index,'team_1']].loc[index]
        except:
            continue
        break
    if index == veto_df.index[-1]:
        complete = 1
    else:
        complete = 0
        
    if complete == 0:
        new_data = veto_df.loc[index+1:]
    else:
        new_data = 0
    return index,complete, new_data
roc_elo = 0.65
roc_diff = 0.3
div = 2
reset_teams_dict = 1

#def compute_map_ratings(teams_dict, roc_elo, roc_diff, div, new):

#this variable is only useful when there already is a trained model.
if reset_teams_dict == 0:
    last_index,complete, new_data = get_last_index()

"""if complete == 1:
    return teams_dict"""

repeat_columns = maps + ['diff_1','diff_else']

if reset_teams_dict == 1:
    teams_dict_params = {'veto_df':veto_df,
                        'start_elo':3,
                        'start_diff':0.5,
                        'span':8,
                        'relevant_teams':relevant_teams}
    teams_dict, teams_curr_row = create_teams_dict(**teams_dict_params)
    index_list = veto_df.index
else:
    index_list = new_data.index


for match in index_list:
    team_1, team_2 = veto_df.loc[match,['team_1','team_2']].values

    match_row = veto_df.loc[match]

    if reset_teams_dict == 1:
        team_1_prev_match = teams_dict[team_1].iloc[teams_curr_row[team_1]]
        team_2_prev_match = teams_dict[team_2].iloc[teams_curr_row[team_2]]

        teams_curr_row[team_1] += 1
        teams_curr_row[team_2] += 1

        teams_dict[team_1].loc[match,repeat_columns] = team_1_prev_match[repeat_columns]
        teams_dict[team_2].loc[match,repeat_columns] = team_2_prev_match[repeat_columns]

        system = teams_dict[team_1].loc[match,'system']
    else:
        team_1_prev_match = teams_dict[team_1].iloc[-1]
        team_2_prev_match = teams_dict[team_2].iloc[-1]

        teams_dict[team_1] = teams_dict[team_1].append(pd.DataFrame([team_1_prev_match],index=[match]))
        teams_dict[team_2] = teams_dict[team_2].append(pd.DataFrame([team_2_prev_match],index=[match]))

        system = match_row.system

    team_1_diff_1 = team_1_prev_match.diff_1
    team_2_diff_1 = team_2_prev_match.diff_1

    team_1_diff_else = team_1_prev_match.diff_else
    team_2_diff_else = team_2_prev_match.diff_else

    team_1_order = first_team_order[system]
    team_2_order = second_team_order[system]

    count_1, count_2 = 1,1

    for i_pick, pick in enumerate(system):

        if pick == '1' and count_1 == 1:
            current = 't1_removed_1'
            count_1 +=1
            _map = match_row[current]

            if np.isfinite(team_1_order[i_pick]):
                team_1_delta_elo = (1/(1+10**((team_1_prev_match[_map]-team_1_order[i_pick])/div))-0.5)*roc_elo
            else:
                team_1_delta_elo = 0

            if np.isfinite(team_2_order[i_pick]):
                team_2_delta_elo = (1/(1+10**((team_2_prev_match[_map]-team_2_order[i_pick])/div))-0.5)*roc_elo
            else:
                team_2_delta_elo = 0

            team_1_delta_teams = (1/(1+10**((team_1_prev_match[_map]-team_2_prev_match[_map])/div))-0.5)*roc_diff

            teams_dict[team_1].loc[match,'diff_1'] += team_1_delta_teams

            teams_dict[team_1].loc[match,_map] += team_1_delta_elo + team_1_delta_teams*team_1_diff_1
            teams_dict[team_2].loc[match,_map] += team_2_delta_elo



            continue

        if pick == '2' and count_2 == 1:
            current = 't2_removed_1'
            count_2 +=1
            _map = match_row[current]

            if np.isfinite(team_1_order[i_pick]):
                team_1_delta_elo = (1/(1+10**((team_1_prev_match[_map]-team_1_order[i_pick])/div))-0.5)*roc_elo
            else:
                team_1_delta_elo = 0

            if np.isfinite(team_2_order[i_pick]):
                team_2_delta_elo = (1/(1+10**((team_2_prev_match[_map]-team_2_order[i_pick])/div))-0.5)*roc_elo
            else:
                team_2_delta_elo = 0

            team_2_delta_teams = (1/(1+10**((team_2_prev_match[_map]-team_1_prev_match[_map])/div))-0.5)*roc_diff

            teams_dict[team_2].loc[match,'diff_1'] += team_2_delta_teams

            teams_dict[team_1].loc[match,_map] += team_1_delta_elo
            teams_dict[team_2].loc[match,_map] += team_2_delta_elo + team_2_delta_teams*team_2_diff_1



            continue

        if pick == '1':
            current = 't1_removed_'+str(count_1)
            count_1 += 1

        elif pick == '2':
            current = 't2_removed_'+str(count_2)
            count_2 += 1

        elif pick == '3':
            current = 't1_picked_1'

        elif pick == '4':
            current = 't2_picked_1'

        elif pick == '5':
            current = 'left_over'

        _map = match_row[current]

        if np.isfinite(team_1_order[i_pick]):
            team_1_delta_elo = (1/(1+10**((team_1_prev_match[_map]-team_1_order[i_pick])/div))-0.5)*roc_elo
        else:
            team_1_delta_elo = 0

        if np.isfinite(team_2_order[i_pick]):
            team_2_delta_elo = (1/(1+10**((team_2_prev_match[_map]-team_2_order[i_pick])/div))-0.5)*roc_elo
        else:
            team_2_delta_elo = 0 

        team_1_delta_teams = (1/(1+10**((team_1_prev_match[_map]-team_2_prev_match[_map])/div))-0.5)*roc_diff
        team_2_delta_teams = -team_1_delta_teams

        if (pick=='1') or (pick=='3'):
            teams_dict[team_1].loc[match,'diff_else'] += team_1_delta_teams

            teams_dict[team_1].loc[match,_map] += team_1_delta_elo + team_1_delta_teams*team_1_diff_else
            teams_dict[team_2].loc[match,_map] += team_2_delta_elo



        elif (pick=='2') or (pick=='4'):
            teams_dict[team_2].loc[match,'diff_else'] += team_2_delta_teams

            teams_dict[team_1].loc[match,_map] += team_1_delta_elo
            teams_dict[team_2].loc[match,_map] += team_2_delta_elo + team_2_delta_teams*team_2_diff_else


    teams_dict[team_1].loc[match,'date'] = match_row.date
    teams_dict[team_2].loc[match,'date'] = match_row.date

    teams_dict[team_1].loc[match,'opponent'] = match_row.team_2
    teams_dict[team_2].loc[match,'opponent'] = match_row.team_1

    teams_dict[team_1].loc[match,maps] = np.clip(teams_dict[team_1].loc[match,maps],1.001,4.999)
    teams_dict[team_2].loc[match,maps] = np.clip(teams_dict[team_2].loc[match,maps],1.001,4.999)

    teams_dict[team_1].loc[match,'diff_1'] = np.clip(teams_dict[team_1].loc[match,'diff_1'],0,1)
    teams_dict[team_2].loc[match,'diff_1'] = np.clip(teams_dict[team_2].loc[match,'diff_1'],0,1)

    teams_dict[team_1].loc[match,'diff_else'] = np.clip(teams_dict[team_1].loc[match,'diff_else'],0,1)
    teams_dict[team_2].loc[match,'diff_else'] = np.clip(teams_dict[team_2].loc[match,'diff_else'],0,1)
        
#    return teams_dict
def remove_not_in_map_pool(teams_dict, maps, picks, relevant_teams):
    matches_map = {}
    for _map in maps:
        matches_map[_map] = veto_df[picks][~veto_df[picks][veto_df[picks]==_map].any(axis=1)].index
        for team in relevant_teams:
            teams_dict[team].loc[teams_dict[team].index.isin(matches_map[_map]),_map] = np.NaN
            
    return teams_dict
teams_dict = remove_not_in_map_pool(teams_dict, maps, picks, relevant_teams)
teams_dict['Astralis'].tail()
"""import pickle
pickle.dump(teams_dict, open( "teams_dict_maps.p", "wb" ) )"""
def Markov_Chain(team1_probs_1,team2_probs_1,team1_probs,team2_probs,team1_inv_probs,team2_inv_probs,team_order,n):
        
    def Reshape(array,level,inverted):
        
        if level == 1:
            if inverted == 0:
                new_shape = [n,1]
            else:
                new_shape = [1,n]
        elif level == 2:
            if inverted == 0:
                new_shape = [n,n,1]
            else:
                new_shape = [n,1,n]
        elif level == 3:
            if inverted == 0:
                new_shape = [n,n,n,1]
            else:
                new_shape = [n,n,1,n]
        elif level == 4:
            if inverted == 0:
                new_shape = [n,n,n,n,1]
            else:
                new_shape = [n,n,n,1,n]
        elif level == 5:
            if inverted == 0:
                new_shape = [n,n,n,n,n,1]
            else:
                new_shape = [n,n,n,n,1,n]
        elif level == 6:
            if inverted == 0:
                new_shape = [n,n,n,n,n,n,1]
            else:
                new_shape = [n,n,n,n,n,1,n]
        
        return array.reshape(new_shape)
    
    def Branch_probs(probs, level):
        return Reshape(probs,level,1)/(1-Reshape(probs,level,0))
    
    def Remove_diagonal(probs, level):
        if level == 1:
            for _map in range(n):
                probs[_map,_map] = 0
        elif level == 2:
            for _map in range(n):
                probs[:,_map,_map] = 0
        elif level == 3:
            for _map in range(n):
                probs[:,:,_map,_map] = 0
        elif level == 4:
            for _map in range(n):
                probs[:,:,:,_map,_map] = 0
        elif level == 5:
            for _map in range(n):
                probs[:,:,:,:,_map,_map] = 0
        elif level == 6:
            for _map in range(n):
                probs[:,:,:,:,:,_map,_map] = 0
        
        return probs
    
    def Weight_branched_probs(probs,old_probs,level):
        return probs*Reshape(old_probs,level,0)

    def Get_final_probs(level, probs_matrix_list, probs_matrix_weighted):
        
        final_probs = probs_matrix_weighted.sum(axis=level-1)
       
        for i in range(level-1,0,-1):
            final_probs = (final_probs * Reshape(probs_matrix_list[i-1], i, 0)).sum(axis=i-1)
            
        return final_probs
 
    def Get_probs_list():
        
        probs_matrix_list_1 = [team1_probs]
        probs_matrix_list_2 = [team2_probs]
        probs_matrix_list_3 = [team1_inv_probs]
        probs_matrix_list_4 = [team2_inv_probs]
        
        probs_matrix_list_1_1 = [team1_probs_1]
        probs_matrix_list_2_1 = [team2_probs_1]
        
        probs_list = [team1_probs_1]
        current_probs_matrix_list = probs_matrix_list_1_1
        probs_matrix_list = [team1_probs_1]
        
        probs_matrix_1_1 = Branch_probs(probs_matrix_list_1_1[0],1)
        probs_matrix_1_1 = Remove_diagonal(probs_matrix_1_1,1)
        probs_matrix_list_1_1.append(probs_matrix_1_1)
        
        cte_2_1 = 0
        cte_3 = 0
        cte_4 = 0
            
        for level in range(1,n):
            probs_matrix_1 = Branch_probs(probs_matrix_list_1[level-1],level)
            probs_matrix_1 = Remove_diagonal(probs_matrix_1,level)
            probs_matrix_list_1.append(probs_matrix_1)
            
            probs_matrix_2 = Branch_probs(probs_matrix_list_2[level-1],level)
            probs_matrix_2 = Remove_diagonal(probs_matrix_2,level)
            probs_matrix_list_2.append(probs_matrix_2)
            
            if cte_3 == 0:
                probs_matrix_3 = Branch_probs(probs_matrix_list_3[level-1],level)
                probs_matrix_3 = Remove_diagonal(probs_matrix_3,level)
                probs_matrix_list_3.append(probs_matrix_3)
            if cte_4 == 0:
                probs_matrix_4 = Branch_probs(probs_matrix_list_4[level-1],level)
                probs_matrix_4 = Remove_diagonal(probs_matrix_4,level)
                probs_matrix_list_4.append(probs_matrix_4)
            if cte_2_1 == 0:
                probs_matrix_2_1 = Branch_probs(probs_matrix_list_2_1[level-1],level)
                probs_matrix_2_1 = Remove_diagonal(probs_matrix_2_1,level)
                probs_matrix_list_2_1.append(probs_matrix_2_1)            
            
            if (team_order[level] == '2') and (cte_2_1==0):
                probs_matrix_weighted = Weight_branched_probs(probs_matrix_list_2_1[level],current_probs_matrix_list[level-1],level)
                current_probs_matrix_list = probs_matrix_list_2_1
                probs_matrix_list.append(probs_matrix_list_2_1[level])
                cte_2_1 = 1
            elif team_order[level] == '1':
                probs_matrix_weighted = Weight_branched_probs(probs_matrix_list_1[level],current_probs_matrix_list[level-1],level)
                current_probs_matrix_list = probs_matrix_list_1
                probs_matrix_list.append(probs_matrix_list_1[level])
            elif team_order[level] == '2':
                probs_matrix_weighted = Weight_branched_probs(probs_matrix_list_2[level],current_probs_matrix_list[level-1],level)
                current_probs_matrix_list = probs_matrix_list_2
                probs_matrix_list.append(probs_matrix_list_2[level])
            elif team_order[level] == '3':
                probs_matrix_weighted = Weight_branched_probs(probs_matrix_list_3[level],current_probs_matrix_list[level-1],level)
                current_probs_matrix_list = probs_matrix_list_3
                probs_matrix_list.append(probs_matrix_list_3[level])
                cte_3 = 1
            elif team_order[level] == '4':
                probs_matrix_weighted = Weight_branched_probs(probs_matrix_list_4[level],current_probs_matrix_list[level-1],level)
                current_probs_matrix_list = probs_matrix_list_4
                probs_matrix_list.append(probs_matrix_list_4[level])
                cte_4 = 1
            elif team_order[level] == '5':
                probs_matrix_weighted = Weight_branched_probs(probs_matrix_list_1[level],current_probs_matrix_list[level-1],level)
                current_probs_matrix_list = probs_matrix_list_1
                probs_matrix_list.append(probs_matrix_list_1[level])
            
            probs = Get_final_probs(level, probs_matrix_list, probs_matrix_weighted)
            probs_list.append(probs)
        return probs_list
    
    return Get_probs_list()

# This chunk of code could be simplified, but I am keeping it like this because I fear any simplification would make the code harder to read.
def get_in_map_pool():
    matches_map = {}
    for _map in maps:
        matches_map[_map] = veto_df[picks][veto_df[picks]==_map].any(axis=1)
    in_map_pool = pd.DataFrame(matches_map)
    return in_map_pool
def get_maps_in_match():
    maps_in_match = {}
    in_map_pool = get_in_map_pool()
    for row in veto_df.index:
        maps_in_match[row] = in_map_pool.loc[row][in_map_pool.loc[row]==True].index
    return maps_in_match
maps_in_match = get_maps_in_match()
div = 1.5
div_diff = 2

probs_list_dict = {}
probs_df_dict = {}
for match in veto_df.index:
    team_1 = veto_df.loc[match,'team_1']
    team_2 = veto_df.loc[match,'team_2']
    count_1,count_2 = 1,1
    
    team_1_df = teams_dict[team_1][maps].loc[match]
    team_1_df = team_1_df[~team_1_df.isna()]
    curr_maps = team_1_df.index.values
    team_1_df = team_1_df.values
    
    team_2_df = teams_dict[team_2][maps].loc[match]
    team_2_df = team_2_df[~team_2_df.isna()].values
    
    team_1_diff_1 = teams_dict[team_1]['diff_1'].loc[match]
    team_2_diff_1 = teams_dict[team_2]['diff_1'].loc[match]
    
    team_1_diff_else = teams_dict[team_1]['diff_else'].loc[match]
    team_2_diff_else = teams_dict[team_2]['diff_else'].loc[match]
    
    system = veto_df.loc[match,'system']
    maps_gone = []
    team_1_diff = team_1_diff_1
    team_2_diff = team_2_diff_1
    
    delta_elo_teams = team_1_df-team_2_df
    
    """probs_1_1 = 1/((5-team_1_df)**2)*((1+team_1_diff_1*0.5)**delta_elo_teams)
    probs_2_1 = 1/((5-team_2_df)**2)*((1+team_2_diff_1*0.5)**(-delta_elo_teams))
    
    probs_1 = 1/((5-team_1_df)**2)*((1+team_1_diff_else*0.5)**delta_elo_teams)
    probs_2 = 1/((5-team_2_df)**2)*((1+team_2_diff_else*0.5)**(-delta_elo_teams))
    
    probs_3 = 1/((team_1_df-1)**2)*((1+team_1_diff_else*0.5)**(-delta_elo_teams))
    probs_4 = 1/((team_2_df-1)**2)*((1+team_2_diff_else*0.5)**delta_elo_teams)"""
    
    probs_1_1 = 1/((5-team_1_df)**2.5)*((1+team_1_diff_1*0.2)**delta_elo_teams)
    probs_2_1 = 1/((5-team_2_df)**2.5)*((1+team_2_diff_1*0.2)**(-delta_elo_teams))
    
    probs_1 = 1/((5-team_1_df)**2.5)*((1+team_1_diff_else*0.2)**delta_elo_teams)
    probs_2 = 1/((5-team_2_df)**2.5)*((1+team_2_diff_else*0.2)**(-delta_elo_teams))
    
    probs_3 = 1/((team_1_df-1)**2.5)*((1+team_1_diff_else*0.2)**(-delta_elo_teams))
    probs_4 = 1/((team_2_df-1)**2.5)*((1+team_2_diff_else*0.2)**delta_elo_teams)

    
    probs_1_1 = probs_1_1/np.sum(probs_1_1)
    probs_2_1 = probs_2_1/np.sum(probs_2_1)
    probs_1 = probs_1/np.sum(probs_1)
    probs_2 = probs_2/np.sum(probs_2)
    probs_3 = probs_3/np.sum(probs_3)
    probs_4 = probs_4/np.sum(probs_4)
    
    probs_list_dict[match] = Markov_Chain(probs_1_1,probs_2_1,probs_1,probs_2,probs_3,probs_4,system,7)
    probs_df_dict[match] = pd.DataFrame(probs_list_dict[match],columns=maps_in_match[match])
del probs_list_dict
def start_bins():

    got_right_bin = { 0.05:0,
                 0.1:0,
                 0.15:0,
                 0.2:0,
                 0.25:0,
                 0.3:0,
                 0.35:0,
                 0.4:0,
                 0.45:0,
                 0.5:0,
                 0.55:0,
                 0.6:0,
                 0.65:0,
                 0.7:0,
                 0.75:0,
                 0.8:0,
                 0.85:0,
                 0.90:0}
    count_bin = got_right_bin.copy()
    
    return got_right_bin, count_bin
def compare_predictions(veto,system):

    got_right_bin, count_bin = start_bins()

    match_index = veto_df[(veto_df.system==system) & (veto_df.date>'2017-04-14')].index
    
    if system == '1212125':
        chosen_veto_list = ['t1_removed_1','t2_removed_1','t1_removed_2','t2_removed_2',
                            't1_removed_3','t2_removed_3','left_over']
    if system == '1234125':
        chosen_veto_list = ['t1_removed_1','t2_removed_1','t1_picked_1','t2_picked_1',
                            't1_removed_2','t2_removed_2','left_over']
            
    for match in match_index:

        for key in got_right_bin.keys():
            for prob in probs_df_dict[match].loc[veto].values:
                if abs(key-prob) < 0.025:
                    count_bin[key] += 1

        prob = probs_df_dict[match].loc[veto,veto_df[chosen_veto_list[veto]].loc[match]]
        for key in got_right_bin.keys():
            if abs(key-prob) < 0.025:
                got_right_bin[key] += 1


    for key in got_right_bin.keys():
        if count_bin[key] != 0:
            got_right_bin[key]/= count_bin[key]
            
    predictions_table = pd.concat([pd.Series(got_right_bin,name='real'),pd.Series(count_bin,name='count')],axis=1).rename_axis('predicted')
    predictions_table
    
    # inefficient code
        
    return predictions_table
predictions_table = compare_predictions(6, system = '1234125')
predictions_table

predictions_table
error_bin = {}
for predicted, real in zip(predictions_table.index,predictions_table.real.values):
    error_bin[predicted] = str(round((real - predicted)*100,2)) + '%'
# shows the error between percentage of occurrences and predictions:
error_bin