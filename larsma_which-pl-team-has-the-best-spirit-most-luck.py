import pandas as pd

import numpy as np

import sqlite3

import os

import yaml

import re

from collections import Counter

import xml.etree.ElementTree as ET



db = os.path.expanduser('../input/database.sqlite')



conn = sqlite3.connect(db)

conn.row_factory = lambda cursor, row: row[0]

cur = conn.cursor()



season = '2015/2016'

league = 1729

league_str = 'England Premier League'



match_ids = cur.execute("SELECT match_api_id FROM Match WHERE league_id=? AND season=? AND goal IS NOT Null", (league, season,)).fetchall()

team_ids = cur.execute("SELECT home_team_api_id FROM Match WHERE league_id=? AND season=? AND goal IS NOT Null", (league, season,)).fetchall()



all_team_ids = cur.execute("SELECT team_api_id FROM Team ORDER BY team_api_id").fetchall()

all_team_names = cur.execute("SELECT team_long_name FROM Team ORDER BY team_api_id").fetchall()

id_name_dict = dict(zip(all_team_ids, all_team_names))



late_scoring_teams_list = []

total_scoring_teams_list = []

goal_type_list = []

#match_ids = [1987032]



unique_list_teams = list(set(team_ids))

net_late_points_dict = dict(zip(unique_list_teams, [0] * len(unique_list_teams)))





for match_id in match_ids:

    goal_str = cur.execute('SELECT goal FROM Match WHERE match_api_id=?', (match_id, )).fetchall()

    home_team_id = cur.execute('SELECT home_team_api_id FROM Match WHERE match_api_id=?', (match_id, )).fetchall()

    away_team_id = cur.execute('SELECT away_team_api_id FROM Match WHERE match_api_id=?', (match_id, )).fetchall()

    home_team_goal = cur.execute('SELECT home_team_goal FROM Match WHERE match_api_id=?', (match_id, )).fetchall()

    away_team_goal = cur.execute('SELECT away_team_goal FROM Match WHERE match_api_id=?', (match_id, )).fetchall()

    team_ids = [home_team_id[0], away_team_id[0]]

    

    late_scoring_teams_list_match = []

    total_scoring_teams_list_match = []

    

    home_team_points = max(np.sign(home_team_goal[0] - away_team_goal[0])*2+1, 0)

    away_team_points = max(np.sign(away_team_goal[0] - home_team_goal[0])*2+1, 0)

    

    root = ET.fromstring(goal_str[0])



    for goal in root:

        elapsed = int(goal.find('elapsed').text)

        goal_type = goal.find('goal_type').text

        team = int(goal.find('team').text)



        if goal_type != 'dg' and goal_type != 'npm' and goal_type != 'rp':

            if elapsed > 85:

                if goal_type == 'o':

                    scoring_team_id = team_ids[:]

                    scoring_team_id.remove(team)

                    late_scoring_teams_list = late_scoring_teams_list + scoring_team_id

                    late_scoring_teams_list_match = late_scoring_teams_list_match + scoring_team_id

                else:

                    late_scoring_teams_list = late_scoring_teams_list + [team]

                    late_scoring_teams_list_match = late_scoring_teams_list_match + [team]

            if goal_type == 'o':

                scoring_team_id = team_ids[:]

                scoring_team_id.remove(team)

                total_scoring_teams_list = total_scoring_teams_list + scoring_team_id

                total_scoring_teams_list_match = total_scoring_teams_list_match + scoring_team_id

            else:

                total_scoring_teams_list = total_scoring_teams_list + [team]

                total_scoring_teams_list_match = total_scoring_teams_list_match + [team]

    

    late_goal_dict_temp = Counter(late_scoring_teams_list_match)

    total_goal_dict_temp = Counter(total_scoring_teams_list_match)

       

    home_team_goal_late = late_goal_dict_temp[home_team_id[0]]

    away_team_goal_late = late_goal_dict_temp[away_team_id[0]]

    

    home_team_goal_early = home_team_goal[0] - home_team_goal_late

    away_team_goal_early = away_team_goal[0] - away_team_goal_late

    

    home_team_points_early = max(np.sign(home_team_goal_early - away_team_goal_early)*2+1, 0)

    away_team_points_early = max(np.sign(away_team_goal_early - home_team_goal_early)*2+1, 0)

    

    home_team_goal_total = total_goal_dict_temp[home_team_id[0]]

    away_team_goal_total = total_goal_dict_temp[away_team_id[0]]    

    

    late_goal_dict_temp = {}

    total_goal_dict_temp = {}

    

    if home_team_goal_total != home_team_goal[0] or away_team_goal_total != away_team_goal[0]:

        raise ValueError('Total goals constructed from incidents does not match end result')

        

    net_late_points_dict[home_team_id[0]] = net_late_points_dict[home_team_id[0]] + (home_team_points-home_team_points_early)

    net_late_points_dict[away_team_id[0]] = net_late_points_dict[away_team_id[0]] + (away_team_points-away_team_points_early)



        

conn.commit()

conn.close()

        
import operator



late_goal_dict_temp = Counter(late_scoring_teams_list)

late_goal_dict = {}



for k, v in late_goal_dict_temp.items():

    late_goal_dict[id_name_dict[k]] = v

    

total_goal_dict_temp = Counter(total_scoring_teams_list)

total_goal_dict = {}



for k, v in total_goal_dict_temp.items():

    total_goal_dict[id_name_dict[k]] = v    



sorted_late_goal = sorted(late_goal_dict.items(), key=operator.itemgetter(1), reverse=True)

late_goal_dict = dict(sorted_late_goal)   



sorted_total_goal = sorted(total_goal_dict.items(), key=operator.itemgetter(1), reverse=True)

total_goal_dict = dict(sorted_total_goal)  
import matplotlib.pyplot as plt



dict_to_plot = late_goal_dict



labels, values = zip(*dict_to_plot.items())





indexes = np.arange(len(labels))

width = 0.5





plt.bar(indexes, values, width)

plt.xticks(indexes, labels, rotation = 90)

#plt.xlabel('Teams', fontweight='bold')

plt.ylabel('# of Late Goals', fontweight='bold')

plt.suptitle('# of Late Goals By Team', fontweight='bold')

plt.title(' - '.join([league_str.split(' ', 1)[1], season]))

plt.show()
import matplotlib.pyplot as plt



normalized_late_goals = {}



for k, v in late_goal_dict.items():

    normalized_late_goals[k] = late_goal_dict[k]/total_goal_dict[k]



sorted_normalized_goal = sorted(normalized_late_goals.items(), key=operator.itemgetter(1), reverse=True)

normalized_late_goals_sorted = dict(sorted_normalized_goal) 

    

dict_to_plot = normalized_late_goals_sorted



labels, values = zip(*dict_to_plot.items())





indexes = np.arange(len(labels))

width = 0.5





plt.bar(indexes, values, width)

plt.xticks(indexes, labels, rotation = 90)

#plt.xlabel('Teams', fontweight='bold')

plt.ylabel('% - Late Goals', fontweight='bold')

plt.suptitle('% - Late Goals By Team', fontweight='bold')

plt.title(' - '.join([league_str.split(' ', 1)[1], season]))

plt.show()
import matplotlib.pyplot as plt





net_late_points_dict_renamed = {}



for k, v in net_late_points_dict.items():

    net_late_points_dict_renamed[id_name_dict[k]] = v





net_late_points_sorted = sorted(net_late_points_dict_renamed.items(), key=operator.itemgetter(1), reverse=True)

net_late_points_sorted_dict = dict(net_late_points_sorted) 

    

dict_to_plot = net_late_points_sorted_dict

    

labels, values = zip(*dict_to_plot.items())





indexes = np.arange(len(labels))

width = 0.5





plt.bar(indexes, values, width)

plt.xticks(indexes, labels, rotation = 90)

#plt.xlabel('Teams', fontweight='bold')

plt.ylabel('# of Net Points', fontweight='bold')

plt.suptitle('# of Net Points after 85min', fontweight='bold')

plt.title(' - '.join([league_str.split(' ', 1)[1], season]))

plt.show()