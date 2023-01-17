import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter # counting instances
import re # parsing play descriptions
import matplotlib.pyplot as plt # data visualizations
from scipy.stats import ttest_ind # signficance test
import os
print(os.listdir("../input"))
# Punt Play Information
play_info_pd = pd.read_csv("../input/play_information.csv")
play_info_pd.head()
play_info_pd.Play_Type.unique() # All the plays are punts
# Concussion (ccus) Information
ccus_review_pd = pd.read_csv("../input/video_review.csv")
ccus_review_pd.head()
# Other files to consider
video_replay = pd.read_csv("../input/video_footage-injury.csv")
# Count varying activities
player_activity_dict = Counter(ccus_review_pd.Player_Activity_Derived)
partner_activity_dict = Counter(ccus_review_pd.Primary_Partner_Activity_Derived)
ff_dict = Counter(ccus_review_pd.Friendly_Fire)
impact_dict = Counter(ccus_review_pd.Primary_Impact_Type)
print("Concussions occur on",round(100*ccus_review_pd.shape[0] / play_info_pd.shape[0],2),"% of punt plays") 
print("\nConcussed Players Activity:")
for activity in player_activity_dict.keys():
    print(activity+':',round(100*player_activity_dict[activity]/ccus_review_pd.shape[0],2), '%')
print("\nConcussed Partners Activity:")
for activity in partner_activity_dict.keys():
    if pd.isnull(activity):
        continue
    print(activity+':',round(100*partner_activity_dict[activity]/ccus_review_pd.shape[0],2), '%')
print('\nFriendly Fire Concussions occur on', round(100*ff_dict['Yes']/ccus_review_pd.shape[0],2), '% of concussions')
print('\nImpact Area:')
for area in impact_dict.keys():
    print(area+':',round(100*impact_dict[area]/ccus_review_pd.shape[0],2), '%')
player_pos_pd = pd.read_csv('../input/player_punt_data.csv').drop(['Number'], axis=1)
player_pos_pd.head()
punt_pos_pd = pd.read_csv('../input/play_player_role_data.csv')
punt_pos_pd.head()
video_replay = pd.read_csv("../input/video_footage-injury.csv")
ccus_positions_pd = ccus_review_pd.join(player_pos_pd.set_index('GSISID'), on='GSISID', how='left').drop_duplicates()
ccus_positions_pd.head()
ccus_both_positions_pd = pd.merge(ccus_positions_pd, punt_pos_pd,  how='left', on=['GSISID','GameKey','PlayID','Season_Year'])
ccus_both_positions_pd.head()
# Count concussions per position
real_pos_dict = Counter(ccus_both_positions_pd.Position)
punt_pos_dict = Counter(ccus_both_positions_pd.Role)
coverage_pos = ['GL','PLW','PLT','PLG','PLS','PRG','PRT','PRW','PC','PPR','P','GR']
return_pos = ['VR','PDR1','PDR2','PDR3','PDL3','PDL2','PDL1','VL','PLR','PLM','PLL','PFB','PR']
coverage_ccus = ccus_both_positions_pd[ccus_both_positions_pd.Role.isin(coverage_pos)].shape[0]
return_ccus = ccus_both_positions_pd[ccus_both_positions_pd.Role.isin(return_pos)].shape[0]
print("Concussed Players Off/Def Position:")
for pos in real_pos_dict.keys():
    print(pos+':',round(100*real_pos_dict[pos]/ccus_review_pd.shape[0],2), '%')
print("\nConcussed Players Punt Position:")
for pos in punt_pos_dict.keys():
    print(pos+':',round(100*punt_pos_dict[pos]/ccus_review_pd.shape[0],2), '%')
print(round(100*coverage_ccus/ccus_review_pd.shape[0],2),'% concussions on coverage,',round(100*return_ccus/ccus_review_pd.shape[0],2),'% concussions on return')
video_replay_pd = pd.read_csv("../input/video_footage-injury.csv")[['gamekey','playid','season','PREVIEW LINK (5000K)']]
video_replay_pd.columns = ['GameKey','PlayID','Season_Year','Video']
video_replay_pd.head()
ccus_videos_pd = pd.merge(ccus_both_positions_pd, video_replay_pd,  how='left', on=['GameKey','PlayID','Season_Year'])
ccus_videos_pd.head()
#build dic bc I'm lazy
# count = 0
# for role in ccus_videos_pd.Role:
#     print("'"+role+"':",count,",")
#     count += 1
action_dict = {
'PLW': 'Tackling PR, foot/ground to head' ,
'GL': 'Blindside block right',
'GR': 'Diving for fumble on muff' ,
'PRT': 'Group Tackling PR' ,
'PRT': 'Blindside block right' ,
'PRW': 'H2H Block on line' ,
'VR': 'head down block' ,
'PFB': 'Pair of blockers run into' ,
'PR': 'tackling during return' ,
'PLG': 'blocked, head to ground near line',
'PLG': 'pair of blockers run into during pursuit' ,
'PRG': 'tackling PR' ,
'PR': 'big hit during return' ,
'P': 'tackled' ,
'PLW': 'chop block knee to head' ,
'GL': 'blocked into PR' ,
'PLG': 'missed tackle' ,
'GL': 'pair of blockers run into',
'GL': 'head to body tackle' ,
'PRG': 'blocked at the line' ,
'PLT': 'blocked chasing PR' ,
'PLG': 'blocked at the line' ,
'PPR': 'group tackle' ,
'PLS': 'blindside block' ,
'PLT': 'H2H tackle' ,
'PR': 'PR tackle' ,
'PLW': 'tackle to ground on line' ,
'PDR1': 'blocking for PR' ,
'PRG': 'missed tackle, FF knee to head' ,
'PR': 'big hit on return' ,
'PDL2': 'H2H throwing big block' ,
'PLL': 'H2H throwing big block' ,
'PR': 'hit on return' ,
'PRW': 'blocking at line' ,
'PLS': 'H2H during tackle' ,
'PLW': 'H2H during block at line' ,
'PRG': 'ran into pair of blockers during tackle' ,
}
# Need Player number so I can follow them on video replay...
# ... merging causes duplicate rows so quick fix is to work with two tables
player_pos_pd = pd.read_csv('../input/player_punt_data.csv')
player_pos_pd.head()
ccus_videos_pd.head()
# Manually iterate through each video to check film for actions
# number = player_pos_pd[player_pos_pd.GSISID == ccus_videos_pd.loc[i].GSISID].Number
# if number.shape[0] > 1:
#     number = list(number)[0]
# else:
#     number = number.item()
# print(ccus_videos_pd.loc[i].Video, ccus_videos_pd.loc[i].Role, number)

# i = i + 1
play_info_pd.head()
ccus_videos_pd.head()
no_return_str_list = ['fair catch','Touchback', 'out of bounds', 'BLOCKED', 'No Play', 'downed by', 
                     'Delay of Game', 'pass', 'False Start', 'Aborted', 'Fake punt']
# How often are punts returned/not returned (return count) and build list of not_returned to compare to ccuss play list
no_return_play_strings = []
return_count = 0
i = 0
while i < len(play_info_pd.PlayDescription):
    play_string = play_info_pd.loc[i].PlayDescription
    return_flag = True
    for phrase in no_return_str_list:
        if phrase in play_string:
            no_return_play_strings.append(play_string)
            return_flag = False
            break
    if return_flag:
        if ('Delay of Game' not in play_string) or ('pass' not in play_string):
            return_count = return_count + 1
    i = i + 1
# How many concussions happened on no return plays
no_return_plays_pd = play_info_pd[play_info_pd.PlayDescription.isin(no_return_play_strings)]
no_return_ccus_pd = pd.merge(ccus_videos_pd,no_return_plays_pd, on=['Season_Year','GameKey','PlayID'], how='inner')
no_return_ccus_count = no_return_ccus_pd.shape[0]
return_ccus_count = ccus_videos_pd.shape[0] - no_return_ccus_pd.shape[0]
print('Punts are returned', round(100*return_count / play_info_pd.shape[0],2), '% of the time')
print('Concussions occur on', round(100*return_ccus_count / return_count,2), '% of the time on return punts,')
print('in comparison to',round(100*no_return_ccus_count / no_return_plays_pd.shape[0],2), "% on no return punts")
print('Players are', round((round(100*return_ccus_count / return_count,2)/round(100*no_return_ccus_count / no_return_plays_pd.shape[0],2)),2), 'times more likely to get concussed during a returned punt rather than a non-returned punt')
# Find Return Yards
return_plays_pd = play_info_pd[~play_info_pd.PlayDescription.isin(no_return_play_strings)]
return_yards_list = []
for play_string in return_plays_pd.PlayDescription:
    try:
        # These are edge cases to cut out
        if ('Delay of Game' in play_string) or ('pass' in play_string) or ('False Start' in play_string) or ('Aborted' in play_string):
            continue
        # 0 yard returns in natural language
        elif ('for no gain' in play_string) or ('MUFFS' in play_string):
            yards = 0
        else:
            yards = int(re.findall(r'for (\-*[0-9]*) yard',play_string)[0])
        return_yards_list.append(yards)
    except:
        print(play_string)
        continue
fig, ax = plt.subplots(dpi=150)  
ax.hist(return_yards_list,  color = "#A8122A", bins=100,)
plt.title('Punt Return Yards')
plt.xlabel('Yards after Catch')
plt.ylabel('# of Punts')
plt.ylim(top=450)
plt.xlim([-20,100])
plt.show()
print('Average (Mean) Punt Return Length:',round(np.mean(return_yards_list),2))
print('Median Punt Return Length:', np.median(return_yards_list))
# Find Punt Length
punt_yards_list = []
for play_string in play_info_pd.PlayDescription:
    try:
        # These are edge cases to cut out
        if ('Delay of Game' in play_string) or ('pass' in play_string) or ('False Start' in play_string) or ('Aborted' in play_string):
            continue
        # More edge cases for all punt scenarios
        if ('BLOCKED' in play_string) or ('formation) PENALTY' in play_string):
            continue
        else:
            yards = int(re.findall(r'punts (\-*[0-9]*) yard',play_string)[0])
        punt_yards_list.append(yards)
    except:
#         print(play_string)
        continue #There wasn't a punt on this play because it was a fake (language too board to continue statment)
# TODO: Why is there a spike at 53 for punt returns?
fig, ax = plt.subplots(dpi=150)  
ax.hist(punt_yards_list, color = '#ffe599', bins=75,)
plt.title('Punt Yards')
plt.xlabel('Length of Punt')
plt.ylabel('# of Punts')
plt.xlim([0,100])
plt.show()
print('Average (Mean) Punt Length:',round(np.mean(punt_yards_list),2))
print('Median Punt Length:', np.median(punt_yards_list))
punt_length_pd = play_info_pd

# Find Punt Length
punt_yards_list = []
# Run the same as above but flag all non-returns so that I can cut them post merge
for play_string in play_info_pd.PlayDescription:
    try:
        # These are edge cases to cut out
        if ('Delay of Game' in play_string) or ('pass' in play_string) or ('False Start' in play_string) or ('Aborted' in play_string):
            yards = 300
        # More edge cases for all punt scenarios
        if ('BLOCKED' in play_string) or ('formation) PENALTY' in play_string):
            yards = 300
        else:
            yards = int(re.findall(r'punts (\-*[0-9]*) yard',play_string)[0])
        punt_yards_list.append(yards)
    except:
        # There wasn't a punt on this play because it was a fake (language too board to continue statment)
        punt_yards_list.append(300)
    
    
punt_length_pd['punt_length'] = punt_yards_list
punt_length_pd.head()
ccuss_length_pd = pd.merge(ccus_videos_pd, punt_length_pd, on=['Season_Year','GameKey','PlayID'], how='inner')
ccuss_length_pd.head()
fortyfive_plus_ccuss_count = ccuss_length_pd[ccuss_length_pd.punt_length <= 45].shape[0]
fortyfive_minus_ccuss_count = ccuss_length_pd[ccuss_length_pd.punt_length != 300].shape[0] - fortyfive_plus_ccuss_count
fortyfive_plus_count = punt_length_pd[punt_length_pd.punt_length <= 45].shape[0]
fortyfive_minus_count = punt_length_pd[punt_length_pd.punt_length != 300].shape[0] - fortyfive_plus_count
print('Concussions occur on', round(100*fortyfive_plus_ccuss_count / fortyfive_plus_count,2), '% of the time on punts longer than 45yrds,')
print('in comparison to',round(100*fortyfive_minus_ccuss_count / fortyfive_minus_count,2), "% on punts shorter than 45")
print('Players are', round((round(100*fortyfive_minus_ccuss_count / fortyfive_minus_count,2)/round(100*fortyfive_plus_ccuss_count / fortyfive_plus_count,2)),2), 'times more likely to get concussed during a punt longer than 45yrds')
# Check for significance
ccuss_length_mark_pd = ccuss_length_pd[['Season_Year','GameKey','Week','PlayID','punt_length']]
ccuss_length_mark_pd['marker'] = 1
ccuss_length_mark_pd.head()
joined = pd.merge(punt_length_pd, ccuss_length_mark_pd, on=['Season_Year','GameKey','Week','PlayID','punt_length'], how='left')
no_ccuss_punts = joined[pd.isnull(joined['marker'])][punt_length_pd.columns]
ccuss_punt_length = ccuss_length_mark_pd.punt_length
no_ccuss_punt_length = no_ccuss_punts.punt_length
stat, pvalue = ttest_ind(ccuss_punt_length,no_ccuss_punt_length)
print('The Line of Scrimmage for concussions, averaging around',round(np.mean(ccuss_punt_length),4), ',\nis statistically distinct from the line for non-concussions, averaging around',round(np.mean(no_ccuss_punt_length),2),':', pvalue < 0.05)
play_info_pd.head()
ccus_videos_pd.head()
punt_location = []
for i in range(play_info_pd.shape[0]):
    line_of_scrim = play_info_pd.loc[i].YardLine
    
    clean_line = re.findall(r'(\w+) ([0-9]+)',line_of_scrim)[0]
    
    team = clean_line[0]
    line = int(clean_line[1])
    
    if team != play_info_pd.loc[i].Poss_Team:
        line = 100 - line
    punt_location.append(line)
plt.hist(punt_location, bins=60,)
plt.title('Punt Locations')
plt.xlabel('Line of Scrimmage')
plt.ylabel('# of Punts')
plt.show()
print('Average (Mean) Location:',round(np.mean(punt_location),2))
print('Median Punt Location:', np.median(punt_location))
punt_loc_pd = play_info_pd
punt_loc_pd = punt_loc_pd[['Season_Year','GameKey','Week','PlayID']]
punt_loc_pd['yard_line'] = pd.Series(punt_location)
punt_loc_pd.head()
ccuss_line_pd = pd.merge(ccus_videos_pd, punt_loc_pd, on=['Season_Year','GameKey','PlayID'], how='inner')
ccuss_line = list(ccuss_line_pd.yard_line)
plt.hist(ccuss_line)
plt.title('Concussed Punt Locations')
plt.xlabel('Line of Scrimmage')
plt.ylabel('# of Punts')
plt.show()
print('Average (Mean) Concussed Punt Location:',round(np.mean(ccuss_line),2))
print('Median Concussed Punt Location:', np.median(ccuss_line))
# Run T-Test to see if this is significant
ccuss_line_pd.head()
within_forty_ccuss_count = ccuss_line_pd[ccuss_line_pd.yard_line <= 40].shape[0]
outside_fourty_ccuss_count = ccuss_line_pd.shape[0] - within_forty_ccuss_count
within_fourty_count = punt_loc_pd[punt_loc_pd.yard_line <= 40].shape[0]
outside_fourty_count = punt_loc_pd.shape[0] - within_fourty_count
print('Concussions occur on', round(100*within_forty_ccuss_count / within_fourty_count,2), '% of the time on punts within own 40,')
print('in comparison to',round(100*outside_fourty_ccuss_count / outside_fourty_count,2), "% on punts outside your 40")
print('Players are', round((round(100*within_forty_ccuss_count / within_fourty_count,2)/round(100*outside_fourty_ccuss_count / outside_fourty_count,2)),2), 'times more likely to get concussed during a punt within your own 40')
within_forty_ccuss_count = ccuss_line_pd[ccuss_line_pd.yard_line <= 35].shape[0]
outside_fourty_ccuss_count = ccuss_line_pd.shape[0] - within_forty_ccuss_count
within_fourty_count = punt_loc_pd[punt_loc_pd.yard_line <= 35].shape[0]
outside_fourty_count = punt_loc_pd.shape[0] - within_fourty_count

print('Concussions occur on', round(100*within_forty_ccuss_count / within_fourty_count,2), '% of the time on punts within own 35,')
print('in comparison to',round(100*outside_fourty_ccuss_count / outside_fourty_count,2), "% on punts outside your 35")
print('Players are', round((round(100*within_forty_ccuss_count / within_fourty_count,2)/round(100*outside_fourty_ccuss_count / outside_fourty_count,2)),2), 'times more likely to get concussed during a punt within your own 35')
# Run t test to prove significance 

ccuss_line_mark_pd = ccuss_line_pd[['Season_Year','GameKey','Week','PlayID','yard_line']]
ccuss_line_mark_pd['marker'] = 1
ccuss_line_mark_pd.head()
punt_loc_pd.head()
joined = pd.merge(punt_loc_pd, ccuss_line_mark_pd, on=['Season_Year','GameKey','Week','PlayID','yard_line'], how='left')
no_ccuss_punts = joined[pd.isnull(joined['marker'])][punt_loc_pd.columns]
ccuss_punt_yards = ccuss_line_mark_pd.yard_line
no_ccuss_punt_yards = no_ccuss_punts.yard_line
stat, pvalue = ttest_ind(ccuss_punt_yards,no_ccuss_punt_yards)
print('The Line of Scrimmage for concussions, averaging around',round(np.mean(ccuss_punt_yards),4), ',\nis statistically distinct from the line for non-concussions, averaging around',round(np.mean(no_ccuss_punt_yards),2),':', pvalue < 0.05)
# http://www.espn.com/nfl/statistics/team/_/stat/returning/position/defense
# Average Kickoff Return Length: 22.98yrds
# Kickoff Touchback Length: 25yrds
# Kickoffs into endzone:
# Kickoffs taken out of endzone: 163
# https://profootballtalk.nbcsports.com/2017/10/17/kickoff-returners-keep-taking-the-ball-out-of-the-end-zone-costing-their-teams-yards/; football outsiders

# 2017 Season Kickoff Stats
# Kickoffs are a good yardstick because they have a touchback (25yrds) that is more than the average kickoff return (21.5yrds) 
# (https://www.teamrankings.com/nfl/stat/touchbacks-per-game?date=2018-02-05) and in 2017 kickoffs have a higher concussion rate (0.6%)
# the average plays (0.4%)... as do punts (0.5%) (https://www.youtube.com/watch?time_continue=449&v=t_SsIKgwvz4)

# By Oct 17, 2017 there were 163 return taken out of the endzone (profootballtalk)
# By that date 75 games had been played (wiki)
# There were an average of 4.96 kickoffs per team per game (https://www.teamrankings.com/nfl/stat/kickoffs-per-game?date=2018-02-05)
# and touchbacks account for 2.8 of those kickoffs (https://www.teamrankings.com/nfl/stat/touchbacks-per-game?date=2018-02-05)
# So with 5.6 touchbacks/game, there are currently 420 touchbacks
# Let's build in the assumption that 25% of those touchbacks are unreturnable-- they go out the back of the endzone (420*.75 = 315)
# % of players opting to touchback when the option is available = 1 - (163/(163+315)) = 66% 
# Based on data more than half of all punt returns would benefit from a 10yrd touchback, so lets assume 50% have the real option of touchback
# If 50% of returns have the option and 66% exercise this option, the number of returns would reduce by 33%
# With returns occuring 33% less, returns reduce from 44% to 29.5%, with no returns occuring on 70.5%
# Assuming no change in % of concussions occuring the types of punt plays, this will reduce

print('~New Rule Concussion %:',round(100*((.44*.0105)+(.56*.0016)),2))
print('New Rule Concussion %:',round(100*((.295*.0105)+(.705*.0016)),2))
delta = round(-100*((((.295*.0105)+(.705*.0016)) - ((.44*.0105)+(.56*.0016))) /((.44*.0105)+(.56*.0016))),2)
print('Rule project to result in ',delta,'% reduction in concussion on punt plays')
i = 0
# Manually iterate through each video to check film for actions
# number = player_pos_pd[player_pos_pd.GSISID == ccus_videos_pd.loc[i].GSISID].Number
# if number.shape[0] > 1:
#     number = list(number)[0]
# else:
#     number = number.item()
# print(action_list[i])
# print(ccus_videos_pd.loc[i].Video, ccus_videos_pd.loc[i].Role, number)

# i = i + 1
return_involved = 29
return_irrelevant = 8
print('Concussions directly related to a return occuring:',round(29/(29+8),2),'%')
print('Return Concussions directly related to a return occuring:', round(29/(29+8-no_return_ccus_count),2),'%')
return_plays_pd = play_info_pd[~play_info_pd.PlayDescription.isin(no_return_play_strings)]
muff_count = 0
for play_string in return_plays_pd.PlayDescription:
    if 'MUFF' in play_string:
        muff_count = muff_count + 1
print(muff_count,' total muffed punts (',round(100*muff_count/return_plays_pd.shape[0],2),'%)')
muff_ccuss_pd = ccus_review_pd
muff_ccuss_pd['marker'] = 1
muff_joined = pd.merge(return_plays_pd, muff_ccuss_pd, on=['Season_Year','GameKey','PlayID'], how='left')
muff_ccuss_count = 0
for i in range(muff_joined.shape[0]):
    play_string = muff_joined.loc[i].PlayDescription
    if ('MUFF' in play_string) and (muff_joined.loc[i].marker == 1):
        muff_ccuss_count = muff_ccuss_count + 1
print(muff_ccuss_count,'concussions occured on muffs,')
print(round(100*muff_ccuss_count/muff_count,2),'% chance of concussion on a muffed punt')
        
# How often to concussions happen on fumbles?
fum_ccuss_pd = ccus_review_pd
fum_ccuss_pd['marker'] = 1
fumble_joined = pd.merge(play_info_pd, fum_ccuss_pd, on=['Season_Year','GameKey','PlayID'], how='left')
        
fumble_count = 0
fumble_ccuss_count = 0
for i in fumble_joined.index:
    play_string = fumble_joined.loc[i].PlayDescription
    if ('FUMBLE' in play_string):
        fumble_count = fumble_count + 1
        if fumble_joined.loc[i].marker == 1:
            fumble_ccuss_count = fumble_ccuss_count + 1
print(round(100*fumble_count/fumble_joined.shape[0],2),'% chance of a fumble')
print(fumble_ccuss_count,'concussions occured on fumbles,')
print(round(100*fumble_ccuss_count/fumble_count,2),'% chance of concussion on a fumble punt')
# How often is a kick returned for a touchdown?
td_count = 0
for play_string in return_plays_pd.PlayDescription:
    if 'TOUCHDOWN' in play_string:
        if 'FUMBLE' in play_string: # Check which team scored on fumble
            print(play_string)
            # Both fumbled TDs were for the defense, so don't count
        else:   
            td_count = td_count + 1
print("\n",round(100*td_count/return_plays_pd.shape[0],2),'% chance of returned TD on returned punt')
print(round(100*td_count/play_info_pd.shape[0],2),'% chance of returned TD on all punt')

new_td_amt = td_count/return_plays_pd.shape[0]*(0.66)
print("\n",round(100*new_td_amt,2),'% chance of returned TD on returned punt post rule change')
print(round(100*(td_count*(0.66))/play_info_pd.shape[0],2),'% chance of returned TD on all punt post rule change')

# print('Rule reduces chance of TD by', round(((td_count/play_info_pd.shape[0])-(td_count*(0.66))/play_info_pd.shape[0]))/(td_count/play_info_pd.shape[0])),2))
# How many penalties are there on returned punts?
penalty_count = 0
no_return_pen_count = 0
for play_string in return_plays_pd.PlayDescription:
    if 'PENALTY' in play_string:
        penalty_count = penalty_count + 1
for play_string in no_return_play_strings:
    if 'PENALTY' in play_string:
        no_return_pen_count = no_return_pen_count + 1
print(round(100*penalty_count/return_plays_pd.shape[0],2),'% chance of penalty on returned punt')
print(round(100*no_return_pen_count/len(no_return_play_strings),2),'% chance of penalty on non-returned punt')
play_info_pd.head()
# How many fair catches occur in the redzone?
fc_count = 0
fc_red_count = 0
for play_string in play_info_pd.PlayDescription:
    if 'fair catch' in play_string:
        fc_count = fc_count + 1
    else:
        continue
    yard_line = int(re.findall(r'yards to\s*[A-Z]*\s(-*[0-9]*)',play_string)[0])
    if yard_line <= 20:
        fc_red_count = fc_red_count + 1
print(round(100*fc_red_count/fc_count,2),'% of fair catches occur in redzone')

# How often are kicks within the redzone returned? Beyond that, what areas of the field have highest/lowest return rates?

# Below is the field dict. This will breakdown where punts land on the field, and how often they are fair caught
# field_fc_dict[key] is the yardline
# field_fc_dict[key][0] is the number of fair caught balls
# field_fc_dict[key][1] is the number of punts in that section of field
field_fc_dict = {
    5:[0,0],
    10:[0,0],
    15:[0,0],
    20:[0,0],
    25:[0,0],
    30:[0,0],
    35:[0,0],
    40:[0,0],
    45:[0,0],
    50:[0,0]
}

for play_string in play_info_pd.PlayDescription:
    if ('punt' not in play_string) or ('Touchback' in play_string) or ('BLOCKED' in play_string):
        continue
    try:
        yard_line = int(re.findall(r'yards to\s*[A-Z]*\s(-*[0-9]*)',play_string)[0])
    except:
        if 'punt' in play_string:
#             print(play_string)
            continue
    for field_section in field_fc_dict.keys():
        if (yard_line <= int(field_section)) and (yard_line > int(field_section)-5):
            if 'fair catch' in play_string:
                field_fc_dict[field_section][0] = field_fc_dict[field_section][0] + 1
            field_fc_dict[field_section][1] = field_fc_dict[field_section][1] + 1
            break

print('\nTotal Punts and Fair Catch Percentage every 5 yards:')
for fs in field_fc_dict.keys():
    perc = round(100*field_fc_dict[fs][0]/field_fc_dict[fs][1],2)
    print(fs,'yards:',field_fc_dict[fs][1],'total punts,',field_fc_dict[fs][0],'fair catches (',perc,'%)' )
        
# Graph
x = list(field_fc_dict.keys())
punts = [field_fc_dict[fs][1] for fs in x]
fcs = [field_fc_dict[fs][0] for fs in x]
fig, ax = plt.subplots(dpi=150)    
ax.bar(x,punts, width = -5, label='punts', color='#ffe599',align='edge')
ax.bar(x,fcs, width = -5, label='fair catches', color='#a8122a',align='edge')
plt.xlabel('Yard Line')
plt.ylabel('# of Punts')
plt.xlim([0,50])
for i in range(len(x)):
    perc = str(round(100*fcs[i]/punts[i],2)) + '%'
    ax.text(x[i]-5,fcs[i]+12,perc,color='black',fontweight='bold', size = 8)
plt.title('The number of punts and the number of fair catches \nthat occur across the field; bucketed every 5 yards',fontsize=10)
plt.suptitle('Punt Outcome per Yard Line', y=1.05, fontsize=18)
ax.legend()
plt.show()
# What percent of punts are within the 15? This percent will represent a reduction in solution efficacy
tot_punts = sum([punts - fcs for punts, fcs in zip(punts, fcs)])
red_punts = sum([punts - fcs for punts, fcs in zip(punts[:3], fcs[:3])])
print(round(100*red_punts/tot_punts,2),'% of returned punts occur within the 15 yard line')
