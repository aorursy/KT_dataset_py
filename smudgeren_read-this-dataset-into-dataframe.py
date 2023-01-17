import numpy as np
import pandas as pd
import json
with open('../season17-18/season_stats.json') as f:
    d = json.load(f)
# create a dictionary containing lists
team_table = {'match_id':[]}

# read through original data
for match_id,match_content in d.items():
    for match_stats in match_content.values():
        team_table['match_id'].append(match_id)
        
        for team_details_label,team_details in match_stats['team_details'].items():
            if team_details_label not in team_table.keys():
                team_table[team_details_label] = []
            team_table[team_details_label].append(team_details)
        
        for team_stats_label,team_stats in match_stats['aggregate_stats'].items():
            if team_stats_label not in team_table.keys():
                team_table[team_stats_label] = []

# so far I've created a dictionary containing all the match&team information,
# but no statistic is filled in due to that not all teams have all the data in every matches,
# now I have to deal with missing data:
# filling all statistics with np.nan
for k,v in team_table.items():
    if len(v) == 0:
        team_table[k] = [np.nan] * len(team_table['match_id'])

# indexing by match ID and team ID, if they both matches, fill in the real data
for pos in range(len(team_table['match_id'])):
    for match_id,match_content in d.items():
        for team_id, match_stats in match_content.items():
            for team_stats_label,team_stats in match_stats['aggregate_stats'].items():
                if (team_table['match_id'][pos] == match_id) & (team_table['team_id'][pos] == team_id):
                    team_table[team_stats_label][pos] = team_stats

df = pd.DataFrame(team_table)
df.to_csv('team_match_data.csv')