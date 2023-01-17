# import libararies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

import seaborn as sns
# Data Load (soccer-1 is the root path original dataset was put)

with sqlite3.connect("../input/soccer-1/database.sqlite") as con:

    countries = pd.read_sql("SELECT * from Country", con)

    matches = pd.read_sql("SELECT * from Match", con)

    leagues = pd.read_sql("SELECT * from League", con)

    teams = pd.read_sql("SELECT * from Team", con)

    teams_stats = pd.read_sql("SELECT * from Team_Attributes", con)

    players = pd.read_sql("SELECT * from Player", con)

    player_stats = pd.read_sql("SELECT * from Player_Attributes", con)
# match related information

ind_match     = matches.columns.get_loc("id")

ind_league     = matches.columns.get_loc("league_id")

ind_season     = matches.columns.get_loc("season")

ind_stage      = matches.columns.get_loc("stage")

# home team information

ind_h_team     = matches.columns.get_loc("home_team_api_id")

ind_h_player   = matches.columns.get_loc("home_player_1")

ind_h_player_x = matches.columns.get_loc("home_player_X1")

ind_h_player_y = matches.columns.get_loc("home_player_Y1")

# away team information

ind_a_team     = matches.columns.get_loc("away_team_api_id")

ind_a_player   = matches.columns.get_loc("away_player_1")

ind_a_player_x = matches.columns.get_loc("away_player_X1")

ind_a_player_y = matches.columns.get_loc("away_player_Y1")
# get all player positions in matches

player_match = pd.DataFrame()

for i in range(0, 11):

    h_info = matches.iloc[:,[ind_match, ind_league,ind_season, ind_stage, ind_h_team, ind_h_player + i, ind_h_player_x + i, ind_h_player_y + i]]

    h_info.columns = ['match_id', 'league_id', 'season', 'stage', 'team_api_id', 'player_api_id', 'player_pos_x', 'player_pos_y']

    a_info = matches.iloc[:,[ind_match, ind_league,ind_season, ind_stage, ind_a_team, ind_a_player + i, ind_a_player_x + i, ind_a_player_y + i]]

    a_info.columns = ['match_id', 'league_id', 'season', 'stage', 'team_api_id', 'player_api_id', 'player_pos_x', 'player_pos_y']

    player_match = player_match.append(h_info, ignore_index = True)

    player_match = player_match.append(a_info, ignore_index = True)



# remove records with null player id    

player_match = player_match[pd.notnull(player_match["player_api_id"])]

# update position x for goal keeper

player_match.loc[player_match['player_pos_y'] == 1, 'player_pos_x'] = 5
# PositionReference.csv is the reference file on position based on player's x-,y- axis in court

pos_ref = pd.read_csv('../input/soccer-2/PositionReference.csv')

player_match = pd.merge(player_match, pos_ref, 

                        on=['player_pos_x', 'player_pos_y'], how='left')

player_match = player_match.drop(['role_x', 'role_y'], 1)
# add team name information

player_match_expnd = pd.merge(player_match, teams[['team_api_id', 'team_long_name']], on=['team_api_id'], how='left')



# add league name information

leagues_nm = leagues.rename(index=str, columns={"id": "league_id", "name": "league_name"})

player_match_expnd = pd.merge(player_match_expnd, leagues_nm[['league_id', 'league_name']], on=['league_id'], how='left')



# add player name information

player_match_expnd = pd.merge(player_match_expnd, players[['player_api_id', 'player_name']], on=['player_api_id'], how='left')
# review player match results

player_match_expnd.loc[(player_match_expnd['player_name'] == 'Lionel Messi') & 

                       (player_match_expnd['season'] == '2010/2011') & 

                       (player_match_expnd['role_xy'] == 'LW') 

                      ]
# write and save to csv

player_match_expnd.to_csv('player_match_expnd.csv', index=False)
# get all tags in xml

def getTags(xMLcolumnNm):

    elemList = []

    # iterate each row in the XML column

    for index, row in matches[pd.notnull(matches[xMLcolumnNm])].iterrows():

        # read in xml data

        tree = ET.ElementTree(ET.fromstring(row[xMLcolumnNm]))

        for elem in tree.iter():

            elemList.append(elem.tag) # append tag name



    # remove duplicates of tag names

    return list(set(elemList))
# parse all information from XML column

def parseXMLData(xMLcolumnNm):

    tags = getTags(xMLcolumnNm) # get a list of all tags

       

    tagLists = {} # host all other tags

    pos = []

    otherList = {'match_id':[], 'pos_x':[], 'pos_y':[]} # host match id

    

    for tag in tags:

        tagLists[tag] = [] # initiate tag lists   

        

    for index, row in matches[pd.notnull(matches[xMLcolumnNm])].iterrows():

        game_id = row['id'] # this helps identify match

        # rea-in XML data

        tree = ET.ElementTree(ET.fromstring(row[xMLcolumnNm]))

        root = tree.getroot()  

        

        for event in root.findall('value'):

            otherList['match_id'].append(game_id)

            for tag in tags:

                if(event.find(tag) is None):

                    tagLists[tag].append(None)

                else:

                    tagLists[tag].append(event.find(tag).text) 

                    

            # get position information

            if(event.find('coordinates') is None): 

                pos.append(None)

                pos.append(None)

            else:  

                for value in event.findall("coordinates/value"):

                    pos.append(value.text)

                    

    otherList['pos_y'] = pos[1::2]  # Elements from list1 starting from 1 iterating by 2

    otherList['pos_x'] = pos[0::2]  # Elements from list1 starting from 0 iterating by 2

            

    xmlInfo = {**otherList, **tagLists}

    return pd.DataFrame(xmlInfo)
# get all xml column information

goal_detail = parseXMLData('goal')

shoton_detail = parseXMLData('shoton')

shotoff_detail = parseXMLData('shotoff')

foulcommit_detail = parseXMLData('foulcommit')

card_detail = parseXMLData('card')

cross_detail = parseXMLData('cross')

corner_detail = parseXMLData('corner')

possession_detail = parseXMLData('possession')
# export xml column information to csv

goal_detail.to_csv('goal_detail.csv', index=False)

shoton_detail.to_csv('shoton_detail.csv', index=False)

shotoff_detail.to_csv('shotoff_detail.csv', index=False)

foulcommit_detail.to_csv('foulcommit_detail.csv', index=False)

card_detail.to_csv('card_detail.csv', index=False)

cross_detail.to_csv('cross_detail.csv', index=False)

corner_detail.to_csv('corner_detail.csv', index=False)

possession_detail.to_csv('possession_detail.csv', index=False)