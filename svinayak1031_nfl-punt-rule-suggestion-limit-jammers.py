import numpy as np
import pandas as pd
import os
import seaborn as sns
import gc
import math
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import random
import tqdm
import scipy
from statsmodels.stats.proportion import proportions_ztest
from timeit import timeit
from scipy.stats.stats import pearsonr 
import re

concussion_how = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
player_punt_role = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
player_role = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
player_role.drop_duplicates(['GSISID'],keep='first',inplace=True)
video = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
video_non = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')
play_info = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')

tqdm.tqdm.pandas()
pd.set_option('display.max_columns', 500)

concussion_plays_temp = pd.merge(concussion_how,player_punt_role[['PlayID','GSISID','GameKey','Role']],how='left',on=['PlayID','GSISID','GameKey'])
concussion_plays_temp2 = pd.merge(concussion_plays_temp, player_role[['GSISID','Position']],how='left',on='GSISID')
concussion_plays = concussion_plays_temp2.rename(columns={'Role':'Player Role','Position':'Player Position'})

player_punt_role['GSISID'] = player_punt_role['GSISID'].astype(str)
concussion_plays =  pd.merge(concussion_plays,player_punt_role[['PlayID','GSISID','GameKey','Role']],how='left',left_on = ['PlayID','Primary_Partner_GSISID','GameKey'],right_on=['PlayID','GSISID','GameKey'])
concussion_plays = concussion_plays.rename(columns={'Role':'Partner Role'})
non_subset = pd.read_csv('../input/nflpuntfilesfinal/non_subset_factors (3).csv')
concussion_plays = pd.read_csv('../input/nflpuntfilesfinal/concussion_plays_factors (3).csv')
non_subset['Men In Box'] = non_subset['Men In Box'].mask(non_subset['Men In Box'] > 11, np.nan)
non_concussion_all = pd.read_csv('../input/non-con-all/non_concussion_plays_all (1).csv')
non_concussion_all['Men In Box'] = non_concussion_all['Men In Box'].mask(non_concussion_all['Men In Box'] > 11, np.nan)
non_concussion_all['Men In Box'] = non_concussion_all['Men In Box'].mask(non_concussion_all['Men In Box'] < 4, np.nan)
non_concussion_all['key'] = non_concussion_all['GSISID'].astype(str) + '-'+ non_concussion_all['GameKey'].astype(str) + '-'+ non_concussion_all['PlayID'].astype(str)
non_concussion_all = non_concussion_all[~non_concussion_all['key'].isin(concussion_plays['key'])]

def import_ngs():
    dtypes = {'Season_Year': 'int16',
             'GameKey': 'int64',
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
    
    # import all ngs files
    
    ngs_files = ['NGS-2016-pre.csv',
                 'NGS-2016-reg-wk1-6.csv',
                 'NGS-2016-reg-wk7-12.csv',
                 'NGS-2016-reg-wk13-17.csv',
                 'NGS-2016-post.csv',
                 'NGS-2017-pre.csv',
                 'NGS-2017-reg-wk1-6.csv',
                 'NGS-2017-reg-wk7-12.csv',
                 'NGS-2017-reg-wk13-17.csv',
                 'NGS-2017-post.csv']
    
    df_list = []
    
    for i in ngs_files:
        df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/'+i, usecols=col_names,dtype=dtypes)
        df_list.append(df)
        print(i)
        
    ngs = pd.concat(df_list)
    
    # Delete the dataframe list to release memory
    del df_list
    gc.collect()
    
    # Convert Time to datetime
    ngs['Time'] = pd.to_datetime(ngs['Time'], format='%Y-%m-%d %H:%M:%S')
    
    # See what we have loaded
    
    ngs['Season_Year'] = ngs['Season_Year'].astype('category').cat.codes
    ngs = ngs[~ngs['GSISID'].isna()]
    ngs['GSISID'] = ngs['GSISID'].astype('int32')
    ngs.reset_index(drop=True,inplace=True)
    
    #ngs.to_feather('ngs_fet.feather')
    return ngs

#ngs = import_ngs()
#print('Done.')
def men_in_box(row):
    """
    This function calculates the number of players on the return team considered lined up 'in the box' 
    at the time of the snap. I used being within 10 yards of the long snapper as a proxy for being 
    considered lined up in the box. 
    
    There was one anomalous case that resulted in 20 players being considered 'in the box', so I
    filtered that out manually. Other than that, the results fit within reason and were confirmed visually 
    where possible.   
    """
    gsisid = row['GSISID']
    play = row['PlayID']
    game = row['GameKey']
    
    return_team_pos = [
 'PDL1','PDL2','PDL3','PDL4','PDL5','PDL6',
 'PDM','PDR1','PDR2','PDR3','PDR4','PDR5',
 'PDR6', 'PLL','PLL1','PLL2','PLL3','PLM',
 'PLM1','PLR','PLR1','PLR2','PLR3','PR',
 'VL', 'VLi','VLo','VR','VRi','VRo', 'PFB','PLS']
    
    #get all players on return team
    play_tracking = ngs[(ngs['GameKey'] == game) & (ngs['PlayID'] == play)]
    play_tracking['GSISID'] = play_tracking['GSISID'].astype(str)
    play_tracking = pd.merge(play_tracking,player_punt_role[['PlayID','GSISID','GameKey','Role']],how='left',on=['PlayID','GSISID','GameKey'])
    return_team_tracking = play_tracking[play_tracking['Role'].isin(return_team_pos)]
    return_team_starting_pos = return_team_tracking[return_team_tracking['Event'] == 'ball_snap']
    if len(return_team_starting_pos[return_team_starting_pos['Role'] == 'PLS']) == 0:
        return np.nan
    
    #locate the long snapper & determine in/out status
    x_ls = return_team_starting_pos[return_team_starting_pos['Role'] == 'PLS']['x'].values[0]
    y_ls = return_team_starting_pos[return_team_starting_pos['Role'] == 'PLS']['y'].values[0]
    men_in_box = 0
    for gsid in return_team_starting_pos[return_team_starting_pos['Role'] != 'PLS']['GSISID']:
        player = return_team_starting_pos[return_team_starting_pos['GSISID'] == gsid]
        x_in = (player['x'].values[0] > x_ls - 10.0) & (player['x'].values[0] < x_ls + 10.0)
        y_in = (player['y'].values[0] > y_ls - 10.0) & (player['y'].values[0] < y_ls + 10.0) 
        if x_in and y_in:
            men_in_box += 1
            
    #discard any errors
    if (men_in_box > 11) or (men_in_box < 4):
        return np.nan
    return men_in_box


def get_kick_list(key):
    """
    Takes in the key (created by combining the GameKey and PlayID separated by a hyphen)
    and outputs the list of roles used by the kicking team on that play.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    return_team_pos = [
 'PDL1','PDL2','PDL3','PDL4','PDL5','PDL6',
 'PDM','PDR1','PDR2','PDR3','PDR4','PDR5',
 'PDR6', 'PLL','PLL1','PLL2','PLL3','PLM',
 'PLM1','PLR','PLR1','PLR2','PLR3','PR',
 'VL', 'VLi','VLo','VR','VRi','VRo', 'PFB']
    
    all_players = ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))]
    all_players['GSISID'] = all_players['GSISID'].astype(str)
    all_players = pd.merge(all_players,player_punt_role[['PlayID','GSISID','GameKey','Role']],how='left',on=['PlayID','GSISID','GameKey'])
    all_roles = set(all_players['Role'])
    kick_roles = []
    for role in all_roles:
        if role not in return_team_pos:
            kick_roles.append(role)
    kick_roles = [x for x in kick_roles if type(x) == str]
    return kick_roles


def get_ret_list(key):
    """
    Takes in the play key (created by combining the GameKey and PlayID separated by a hyphen)
    and outputs the list of roles used by the return team on that play.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    return_team_pos = [
 'PDL1','PDL2','PDL3','PDL4','PDL5','PDL6',
 'PDM','PDR1','PDR2','PDR3','PDR4','PDR5',
 'PDR6', 'PLL','PLL1','PLL2','PLL3','PLM',
 'PLM1','PLR','PLR1','PLR2','PLR3','PR',
 'VL', 'VLi','VLo','VR','VRi','VRo', 'PFB']
    
    all_players = ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))]
    all_players['GSISID'] = all_players['GSISID'].astype(str)
    all_players = pd.merge(all_players,player_punt_role[['PlayID','GSISID','GameKey','Role']],how='left',on=['PlayID','GSISID','GameKey'])
    all_roles = set(all_players['Role'])
    ret_roles = []
    for role in all_roles:
        if role in return_team_pos:
            ret_roles.append(role)
    ret_roles = [x for x in ret_roles if type(x) == str]
    return ret_roles
boxes = ['6','7','8','Other']
six = round(100*len(non_concussion_all[non_concussion_all['Men In Box']==6])/len(non_concussion_all['Men In Box'].dropna()),2)
seven = round(100*len(non_concussion_all[non_concussion_all['Men In Box']==7])/len(non_concussion_all['Men In Box'].dropna()),2)
eight = round(100*len(non_concussion_all[non_concussion_all['Men In Box']==8])/len(non_concussion_all['Men In Box'].dropna()),2)
other = 100 - six - seven - eight
freq = [six,seven,eight,other]
graph_df = pd.DataFrame({'Box Size':boxes,'Frequency':freq})

sns.set(style="whitegrid")
plt.style.use('fivethirtyeight');
clrs = ['dodgerblue','dodgerblue','indianred','dodgerblue']
f, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x='Box Size',y='Frequency',data=graph_df,palette = clrs)
plt.xticks(size=20);
plt.yticks(np.arange(0,45,5),size=12)
plt.ylabel('Frequency (%)');
plt.title('Frequency of Box Sizes, All Plays');

boxes = ['6','7','8']
six = round(100*len(concussion_plays[concussion_plays['Men In Box']==6])/len(concussion_plays['Men In Box'].dropna()),2)
seven = round(100*len(concussion_plays[concussion_plays['Men In Box']==7])/len(concussion_plays['Men In Box'].dropna()),2)
eight = round(100*len(concussion_plays[concussion_plays['Men In Box']==8])/len(concussion_plays['Men In Box'].dropna()),2)
freq = [six,seven,eight]
graph_df = pd.DataFrame({'Box Size':boxes,'Frequency':freq})

sns.set(style="whitegrid")
plt.style.use('fivethirtyeight');


f, ax = plt.subplots(figsize=(10, 10))
clrs = ['indianred','dodgerblue','dodgerblue']
sns.barplot(x='Box Size',y='Frequency',data=graph_df,palette = clrs)
plt.xticks(size=20);
plt.yticks(np.arange(0,45,5),size=12)
plt.ylabel('Frequency (%)');
plt.title('Frequency of Box Sizes, Concussion Plays');
def test_significance(concussion_plays, non_concussion_plays):
    count = concussion_plays['Men In Box'].value_counts()[6]
    nobs = len(concussion_plays)
    value = (non_concussion_plays['Men In Box'].value_counts()/len(non_concussion_plays))[6]
    stat, pval = proportions_ztest(count, nobs, value)
    if pval < .05:
        result = """
        Given that the population proportion of instances in which there were six men in the \n\
        box on punts was {pop}, there is only a {pval} probability that the {pct} proportion \n\
        seen in the plays in which concussions occured was drawn from the population \n\
        distribution. Thus, at a 95% confidence level, we reject the null that the sample \n\
        and population means are equal.
        """.format(pop= round(value,6), pval = round(pval,6), pct = round(count/nobs,6))
    else:
        result =    """
        Given that the population proportion of instances in which there were six men in \n\ 
        the box on punts was {pop},there is a {pval} probability that the {pct} proportion \n\
        seen in the plays in which concussions occured was drawn from the population distribution.\n\ 
        Thus, we cannot reject the null that the sample and population means are equal.
        """.format(pop= round(value,6), pval = round(pval,6), pct = round(count/nobs,6))
    print(result)
    return pval

pval = test_significance(concussion_plays,non_concussion_all)
six_gun = non_subset.groupby('Men In Box').mean()['count_gunners'][6]
eight_gun = non_subset.groupby('Men In Box').mean()['count_gunners'][8]

six_jam = non_subset.groupby('Men In Box').mean()['count_corners'][6]
eight_jam = non_subset.groupby('Men In Box').mean()['count_corners'][8]

print("Number of gunners on 6 man box plays: ", round(six_gun,1))
print("Number of gunners on 8 man box plays: ", round(eight_gun,1))
print(" ")
print("Number of jammers on 6 man box plays: ", round(six_jam,1))
print("Number of jammers on 8 man box plays: ", round(eight_jam,1))
fair_six = non_concussion_all.groupby('Men In Box').mean()['Fair Catch'][6]
fair_eight = non_concussion_all.groupby('Men In Box').mean()['Fair Catch'][8]
ret_six = non_subset.groupby('Men In Box').mean()['yards_returned'][6]
ret_eight = non_subset.groupby('Men In Box').mean()['yards_returned'][8]
print("Yards per return, 6-man box: ",ret_six)
print("Yards per return, 8-man box: ",ret_eight)
print(" ")
print(" ")
print(pd.DataFrame(non_subset.groupby('Men In Box').mean()['Average Play Speed']))
print(" ")
print(" ")
time_six = non_subset.groupby('Men In Box').mean()['time_to_punt'][6]
time_eight = non_subset.groupby('Men In Box').mean()['time_to_punt'][8]
print("Average snap-to-punt time, 6-man box: ", time_six)
print("Average snap-to-punt time, 8-man box: ", time_eight)
print(" ")
print(" ")
hang_six = non_concussion_all.groupby('Men In Box').mean()['Hang Time'][6]
hang_eight = non_concussion_all.groupby('Men In Box').mean()['Hang Time'][8]
print("Average punt hang-time, 6-man box: ",hang_six)
print("Average punt hang-time, 8-man box: ",hang_eight)

print(" ")
print(" ")
print("Percent of plays resulting in a fair catch, 6-man box: ",round(100* fair_six, 2))
print("Percent of plays resulting in a fair catch, 8-man box: ",round(100 * fair_eight, 2))
non_subset.groupby('Men In Box').mean()['time_to_punt'][8]
print(" ")
print(" ")
fair_c = len(concussion_plays[concussion_plays['Fair Catch'] == True])/len(concussion_plays)
print("Percent of plays resulting in a fair catch + concussion: ",round(fair_c * 100,2))
def test_significance_fair_catch(concussion_plays, non_concussion_plays):
    count = len(concussion_plays[concussion_plays['Fair Catch'] == True])
    nobs = len(concussion_plays)
    value = len(non_concussion_plays[non_concussion_plays['Fair Catch'] == True])/len(non_concussion_plays)
    stat, pval = proportions_ztest(count, nobs, value)
    if pval < .05:
        result = """
        Given that the population proportion of instances in which a fair catch \n\
        occured on punts was {pop},there is only a {pval} probability that the \n\
        {pct} proportion seen in the plays in which concussions occured was drawn \n\
        from the population distribution. Thus, we reject the null that the sample \n\
        and population means are equal.
        """.format(pop= round(value,6), pval = round(pval,6), pct = round(count/nobs,6))
    else:
        result =    """
        Given that the population proportion of instances in which a fair catch occured on punts was {pop},
        there is a {pval} probability that the {pct} proportion seen in the plays in which concussions occured 
        was drawn from the population distribution. Thus, we cannot reject the null that the sample and population means are equal.
        """.format(pop= round(value,6), pval = round(pval,6), pct = round(count/nobs,6))
    print(result)
    return pval

pval_fc = test_significance_fair_catch(concussion_plays,non_concussion_all)
def get_player_speeds(df):
    """
    Takes in a player's NGS records for a single play and
    returns his average speed after converting to miles per hour.
    """
    dis = df['dis']
    mph = dis * 20.455
    return np.mean(mph)


def get_video_link(game, play):
    """
    Function used to retrieve video link given game and play.
    """
    return video[(video['gamekey'] == game) & (video['playid'] == play)]['PREVIEW LINK (5000K)'].values[0]


def check_fair_catch(key):
    """
    Checks whether or not a given play resulted in a fair catch.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    events = set(ngs[(ngs['PlayID'] == int(playID)) & (ngs['GameKey'] == int(gameID))]['Event'])
    if 'fair_catch' in events:
        return True
    else:
        return False
    
def play_speed_avg(key):
    """
    Calculates the average speed for all players that participated
    in a given play.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    play = ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))][['dis','GSISID']]
    all_play_speeds = []
    all_averages = []
    for gsid in set(play['GSISID']):
        player_play = play[play['GSISID'] == gsid]
        player_avg = get_player_speeds(player_play)
        all_averages.append(player_avg)
    avg = np.mean(all_averages)
    return avg


def play_speed_sd(key):
    """
    Calculates the standard deviation of the  speed for 
    all players that participated in a given play.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    play = ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))][['dis','GSISID']]
    all_play_speeds = []
    all_averages = []
    for gsid in set(play['GSISID']):
        player_play = play[play['GSISID'] == gsid]
        player_avg = get_player_speeds(player_play)
        all_averages.append(player_avg)
    sd = np.std(all_averages)
    return sd


def play_speed_max(key):
    """
    Returns the max speed of the fastest player on a given play.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    play = ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))][['dis','GSISID']]
    all_play_speeds = []
    all_averages = []
    for gsid in set(play['GSISID']):
        player_play = play[play['GSISID'] == gsid]
        player_avg = get_player_speeds(player_play)
        all_averages.append(player_avg)
    mx = max(all_averages)
    return mx


def get_hang_time(row):
    """
    Returns the hang time of the punt on a given play.
    """
    playID = row['PlayID']
    GameKey = row['GameKey']
    gsid = row['GSISID']
    ngs_play = ngs[(ngs['PlayID'] == playID) & (ngs['GameKey'] == GameKey)]
    ngs_play = ngs_play[ngs_play['GSISID'] == gsid]
    punt = ngs_play[ngs_play['Event'] == 'punt']['Time']
    end_events = ['fair_catch','punt_downed','punt_land','punt_muffed','punt_received']
    end_hang = ngs_play[ngs_play['Event'].isin(end_events)]
    if len(end_hang) == 0:
        return np.nan
    elif len(end_hang) > 1:
        end_hang = end_hang[end_hang['Time'] == min(end_hang['Time'])]
        
    try:
        hang_time = end_hang['Time'].values[0] - punt.values[0]
    except:
        return np.nan
    hang_time = int(hang_time)/1000000000
    if hang_time > 6:
        return np.nan
    elif hang_time < 1.5:
        return np.nan
    return hang_time


def plot_player_speed(ngs, play,GSISID):
    """
    Function used to plot a player's velocity throughout a play.
    """
    player_tracking = ngs[(ngs['PlayID'] == play) & (ngs['GSISID'] == GSISID)].sort_values('Time')
    player_tracking['Speed'] = player_tracking.apply(lambda x:get_speed(x,player_tracking),axis=1)
    ax = sns.lineplot(x=range(len(player_tracking)),y='Speed',data=player_tracking, )
    ax.set(xlabel='Time Elapsed', ylabel='MPH');
    plt.show()
    print(get_video_link(player_tracking['GameKey'].iloc[0], play))
    print('Position: ' + player_role[player_role['GSISID'] == GSISID]['Position'].values[0])
    print('Number: ' + player_role[player_role['GSISID'] == GSISID]['Number'].values[0])
    print('Punt Role: ' + concussion_plays[(concussion_plays['GSISID'] == GSISID) & (concussion_plays['PlayID'] == play)]['Player Role'].values[0])
    print('Activity: ' + concussion_plays[(concussion_plays['GSISID'] == GSISID) & (concussion_plays['PlayID'] == play)]['Player_Activity_Derived'].values[0])
    

def count_gunners(roles):
    count = 0
    for role in roles:
        if role[0] == 'G':
            count += 1
    return count

def count_corners(roles):
    count = 0
    for role in roles:
        if role[0] == 'V':
            count += 1
    return count

    
def yards_returned(key):
    """
    Parses the number of yards a returner gained from the play description using regex.
    Any plays involving a penalty or fumble was thrown out.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    play_record = play_info[(play_info['GameKey']==int(gameID)) & (play_info['PlayID']==int(playID))]
    try:
        des = play_record['PlayDescription'].values[0]
    except IndexError:
        des = play_record['PlayDescription']
    play_events = set(ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))]['Event'])
    discard = ['fair_catch','field_goal_attempt','field_goal_play','fumble','out_of_bounds','penalty_flag','punt_blocked','touchback']
    if any(i in play_events for i in discard):
        return np.nan
    elif 'punts' not in des:
        return np.nan
    elif "PENALTY" in des:
        return np.nan
    
    regex_num = re.compile(r"[+-]?\d+(?:\.\d+)?")
    nums = regex_num.findall(des.split('Center')[1])
    if len(nums) != 2:
        return np.nan
    return int(nums[1])

def start_position(key):
    """
    Returns the starting position of a punt. Instead of 
    indicating side of field, wrote all yardage
    out of 100. 99 yards = 99 yards away from scoring a touchdown.
    """
    try:
        keysplit = key.split('-')
        gameID = keysplit[1]
        playID = keysplit[2]
        play_record = play_info[(play_info['GameKey']==int(gameID)) & (play_info['PlayID']==int(playID))]
        try:
            poss = play_record['Poss_Team'].values[0]
            start = play_record['YardLine'].values[0]
        except IndexError:
            poss = play_record['Poss_Team']
            start = play_record['YardLine']
        play_events = set(ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))])
        discard = ['fair_catch','field_goal_attempt','field_goal_play','fumble','out_of_bounds','penalty_flag','punt_blocked','touchback']
        if any(i in play_events for i in discard):
            return np.nan
        start_yard = start.split(' ')[1]
        start_team = start.split(' ')[0]
        if start_team == poss:
            yard_line = (50 - int(start_yard)) + 50
        else:
            yard_line = int(start_yard)
        return yard_line
    except:
        return np.nan
    
def snap_to_punt(key):
    """
    Time taken from the ball snap to the punt execution.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    play = ngs[(ngs['GameKey'] == int(gameID)) & (ngs['PlayID'] == int(playID))][['Event','Time']]
    if 'punt' not in set(play['Event']):
        return np.nan
    start = min(play[play['Event']=='ball_snap']['Time'])
    punt = min(play[play['Event']=='punt']['Time'])
    time_to_punt = punt - start
    return time_to_punt.total_seconds()


def check_fumble(key):
    """
    Returns True if a fumble occured on a play,
    False otherwise.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    events = set(ngs[(ngs['PlayID'] == int(playID)) & (ngs['GameKey'] == int(gameID))]['Event'])
    if 'fumble' in events:
        return True
    else:
        return False
    
    
def check_td(key):
    """
    Returns True if a touchdown was scored on a play,
    False otherwise.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    events = set(ngs[(ngs['PlayID'] == int(playID)) & (ngs['GameKey'] == int(gameID))]['Event'])
    if 'touchdown' in events:
        return True
    else:
        return False
    
def check_block(key):
    """
    Returns True if a punt was blocked,
    False otherwise.
    """
    keysplit = key.split('-')
    gameID = keysplit[1]
    playID = keysplit[2]
    events = set(ngs[(ngs['PlayID'] == int(playID)) & (ngs['GameKey'] == int(gameID))]['Event'])
    if 'punt_blocked' in events:
        return True
    else:
        return False


## calculated columns
"""
concussion_plays['Fair Catch'] = concussion_plays.progress_apply(lambda x:check_fair_catch(x), axis = 1)
concussion_plays['Average Play Speed'] = concussion_plays['key'].progress_apply(lambda x:play_speed_avg(x))
concussion_plays['Std Play Speed'] = concussion_plays['key'].progress_apply(lambda x:play_speed_sd(x))
concussion_plays['Max Play Speed'] = concussion_plays['key'].progress_apply(lambda x:play_speed_max(x))
concussion_plays['Hang Time'] = concussion_plays.progress_apply(lambda x:get_hang_time(x), axis = 1)
concussion_plays['Men In Box'] = concussion_plays.progress_apply(lambda x:men_in_box(x), axis = 1)

non_subset['Average Play Speed'] = non_subset['key'].progress_apply(lambda x:play_speed_avg(x))
non_subset['Std Play Speed'] = non_subset['key'].progress_apply(lambda x:play_speed_sd(x))
non_subset['Max Play Speed'] = non_subset['key'].progress_apply(lambda x:play_speed_max(x))

#add kick roles
concussion_plays['kick_roles'] = concussion_plays['key'].progress_apply(lambda x:get_kick_list(x))
non_subset['kick_roles'] = non_subset['key'].progress_apply(lambda x:get_kick_list(x))
#non_subset_pos = non_subset[non_subset['kick_roles'].str.len() == 11]

#add ret roles
concussion_plays['ret_roles'] = concussion_plays['key'].progress_apply(lambda x:get_ret_list(x))
non_subset['ret_roles'] = non_subset['key'].progress_apply(lambda x:get_ret_list(x))

#count gunners and corners
concussion_plays['count_gunners'] = concussion_plays['kick_roles'].progress_apply(lambda x:count_gunners(x))
non_subset['count_gunners'] = non_subset['kick_roles'].progress_apply(lambda x:count_gunners(x))
concussion_plays['count_corners'] = concussion_plays['ret_roles'].progress_apply(lambda x:count_corners(x))
non_subset['count_corners'] = non_subset['ret_roles'].progress_apply(lambda x:count_corners(x))

non_subset_pos_temp = non_subset[non_subset['kick_roles'].str.len() == 11]
non_subset_pos = non_subset_pos_temp[non_subset_pos_temp['ret_roles'].str.len() == 11]
non_subset['yards returned'] = non_subset['key'].progress_apply(lambda x:yards_returned(x))


non_subset['start_position'] = non_subset['key'].progress_apply(lambda x:start_position(x))
concussion_plays['start_position'] = concussion_plays['key'].progress_apply(lambda x:start_position(x))

concussion_plays['time_to_punt'] = concussion_plays['key'].progress_apply(lambda x:snap_to_punt(x))
non_subset['time_to_punt'] = non_subset['key'].progress_apply(lambda x:snap_to_punt(x))
non_subset['yards returned'] = non_subset['key'].progress_apply(lambda x:yards_returned(x))

non_subset['touchback'] = non_subset['key'].progress_apply(lambda x:check_touchback(x))
concussion_plays['touchback'] = concussion_plays['key'].progress_apply(lambda x:check_touchback(x))

concussion_plays.to_csv('concussion_plays_factors.csv')
non_subset.to_csv('non_subset_factors.csv')
non_subset['yards_returned'] = non_subset['key'].progress_apply(lambda x:yards_returned(x))

#concussion_plays['time_to_punt'] = concussion_plays['key'].progress_apply(lambda x:snap_to_punt(x))
non_subset['time_to_punt'] = non_subset['key'].progress_apply(lambda x:snap_to_punt(x))

non_subset['fumble'] = non_subset['key'].progress_apply(lambda x:check_fumble(x))
non_subset['touchdown'] = non_subset['key'].progress_apply(lambda x:check_td(x))

non_subset['blocked'] = non_subset['key'].progress_apply(lambda x:check_block(x))
"""
concussion_plays.to_csv('concussion_plays_factors.csv')
non_subset.to_csv('non_subset_factors.csv')