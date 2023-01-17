import numpy as np

import pandas as pd

import sqlite3 as sql

import matplotlib.pyplot as plt

import seaborn as sns

import xml.etree.ElementTree as ET

from IPython.display import Image

%matplotlib inline
database = '/kaggle/input/soccer/database.sqlite'

connection = sql.connect(database)
query = '''SELECT * FROM Country'''

df_country = pd.read_sql_query(query, connection)

df_country.head()
df_country.info()
query = '''SELECT * FROM League'''

df_league = pd.read_sql_query(query, connection)

df_league.head()
df_league.info()
query = '''SELECT * FROM Player'''

df_player = pd.read_sql_query(query, connection)

df_player.head()
df_player.shape
df_player.info()
query = '''SELECT * FROM Player_Attributes'''

df_player_attr = pd.read_sql_query(query, connection)

df_player_attr.head()
df_player_attr['player_fifa_api_id'].nunique()
df_player_attr.shape
df_player_attr = None
query = '''SELECT * FROM Team'''

df_team = pd.read_sql_query(query, connection)

df_team.head()
df_team.info()
df_team[df_team['team_fifa_api_id'].isnull()]
query = '''SELECT * FROM Team_Attributes'''

df_team_attr = pd.read_sql_query(query, connection)

df_team_attr.head()
df_team_attr = None
query = '''SELECT * FROM Match'''

df_match = pd.read_sql_query(query, connection)

df_match.head()
# Identify which of the Team IDs is used by taking the first example (team 9987)



df_team[df_team['team_api_id'] == 9987]
df_team[df_team['team_fifa_api_id'] == 9987]
df_match.shape
list(df_match.columns)
# Identify which of the Player IDs is used by taking a random player ID and then 

# check which ID field in the Player table contains it



id = df_match['home_player_1'][1000]



df_player[df_player['player_api_id'] == id]
df_player[df_player['player_fifa_api_id'] == id]
# Join the Country and League tables



df_league = df_league.merge(df_country, on="id", how="inner")

df_league.head(3)
# Drop the country_id column



df_league.drop('country_id', axis=1, inplace=True)



# Rename the columns



df_league = df_league.rename(columns = {'name_x': 'league_name', 'name_y': 'country'})



df_league.head(3)
# Drop the 'id' and the 'player_fifa_api_id' columns from the Player table

df_player.drop(columns = ['id', 'player_fifa_api_id'], inplace=True)



# Rename the player id

df_player = df_player.rename(columns = {'player_api_id': 'player_id'})



df_player.head()
# Drop the 'id' and the 'team_fifa_api_id' columns from the Team table

df_team.drop(columns = ['id', 'team_fifa_api_id'], inplace=True)



# Rename the team id

df_team = df_team.rename(columns = {'team_api_id': 'team_id'})



df_team.head()
# Drop the X positions columns, the betting columns (except the B365 betting odds), and the 'id' column



df_match = df_match.iloc[:, np.r_[1:11, 33:88]]



# Rename the match and the team IDs



df_match = df_match.rename(columns = {'match_api_id': 'match_id', 'home_team_api_id': 'home_team_id', 'away_team_api_id': 'away_team_id'})



df_match.head()
# Join the Match table with the League table

# The relationship is many to 1



df_match = df_match.merge(df_league, left_on="league_id", right_on="id", how="inner")



# Drop the 'country_id', the 'league_id' and the 'id' columns



df_match.drop(columns = ['country_id', 'league_id', 'id'], inplace=True)



df_match.head()
# Include all columns containing the Y coordinate in a list



position_columns = []



for i, col in enumerate(df_match.columns):

    if 'player_Y'in col:

        position_columns.append(col)     
# Transform the position number into the position (goalkeeper, defender, midfielder, forward)



def getFieldPosition(position):

    if position == 1:

        return 'goalkeeper'

    if 2 <= position <= 5:

        return 'defender'

    if 6 <= position <= 8:

        return 'midfielder'

    if 9 <= position <= 11:

        return 'forward'

    return None
# Apply the getFieldPositionFunction to all the columns in the Data Frame that contain the Y coordinate



for col in position_columns:

    df_match[col] = df_match[col].apply(getFieldPosition)
# Rename the position columns from 'home_player_Y1', 'away_player_Y11', etc to 'home_player_1_pos', 'away_player_11_pos', etc

        

def renamePositionColumns(dataframe):

    col_names = []



    for i, col in enumerate(dataframe.columns):

        if 'player_Y'in col:

            parts = col.split('_')

            playerNo = parts[2][1:]

            new_col_name = parts[0] + '_' + parts[1] + '_' + playerNo + '_' + 'pos'

            col_names.append(new_col_name)

        else:

            col_names.append(col)

    

    dataframe.columns = col_names

            

    return dataframe



df_match = renamePositionColumns(df_match)



# Verify if the 'translation' to field position and the change in column names worked  

df_match.tail()
# Drop further columns not needed for the analysis



drop_columns = ['shoton', 'shotoff', 'foulcommit', 'cross', 'possession']

df_match.drop(columns = drop_columns, inplace=True)
# Read XML and extract which players scored the goals

# Display the value of a 'goal' field to analyse the XML format



df_match['goal'][1728]
# Display what the score was in the match



df_match.loc[1728, ['match_id', 'home_team_id', 'away_team_id', 'home_team_goal', 'away_team_goal']]
# Function that extract the IDs of the players that scored



def getPlayersThatScored(xmlInfo):

    if xmlInfo == None:

        return None

    

    scorers = []

    

    root = ET.fromstring(xmlInfo)

    for child in root:

        player = child.find('player1')

        goal_type = child.find('goal_type')

        

        if player != None and goal_type != None:

            # Please check in the next cell why the following line was needed

            if goal_type.text not in ['dg', 'npm', 'o', 'rp']:

                scorers.append(player.text)  

                

    if len(scorers) > 0:

        return ';'.join(scorers)

    else:

        return None
# Apply the 'getPlayersThatScored' function on the 'goal' column of the Data Frame

# The purpose is to have the IDs of the players that scored in the 'goal' column



df_match['scorers'] = df_match['goal'].apply(getPlayersThatScored)



# Verify it is worked on the England Premier League input (as it has less NULLs)

df_match[df_match['league_name'] == 'England Premier League'][['home_team_goal', 'away_team_goal', 'scorers']].head()

#Display the value of a 'card' field to analyse the XML format



df_match['card'][1728]
# Function that parses the 'card' and 'corner' XML blocks and returns the number of cards/corners each team received during the match

# The input argument is a Series that corresponds to each row of the Data Frame on which the function is applied



def getStatsPerTeam(match_info):

    xmlInfo = match_info[0]

    if xmlInfo == None:

        return [None, None]

    

    home_team_stats = 0

    away_team_stats = 0

    

    root = ET.fromstring(xmlInfo)

    for child in root:

        team = child.find('team')

        if team != None:

            if int(team.text) == match_info['home_team_id']:

                home_team_stats = home_team_stats + 1

            elif int(team.text) == match_info['away_team_id']:

                away_team_stats = away_team_stats + 1

                

    return [home_team_stats, away_team_stats]
# Apply the 'getStatsPerTeam' function to store the cards number per team in two new columns

df_match[['cards_home_team', 'cards_away_team']] = df_match[['card', 'home_team_id', 'away_team_id']].apply(getStatsPerTeam, axis = 1, result_type='expand')



# Apply the 'getStatsPerTeam' function to store the number of corners per team in two new columns

df_match[['corners_home_team', 'corners_away_team']] = df_match[['corner', 'home_team_id', 'away_team_id']].apply(getStatsPerTeam, axis = 1, result_type='expand')



# Verify it is worked on the England Premier League input (as it has less NULLs)

df_match[df_match['league_name'] == 'England Premier League'][['cards_home_team', 'cards_away_team', 'corners_home_team', 'corners_away_team']].head()

# Function to transform the score into a concrete result for each team: win, loss, draw



def getResultWinLoss(match_info):

    if match_info['home_team_goal'] > match_info['away_team_goal']:

        home_team_result = 'Win'

        away_team_result = 'Loss'

    elif match_info['home_team_goal'] < match_info['away_team_goal']:

        home_team_result = 'Loss'

        away_team_result = 'Win'

    else: 

        home_team_result = 'Draw'

        away_team_result = 'Draw'

    return [home_team_result, away_team_result]
# Get team result: win, loss, draw



df_match[['home_team_result', 'away_team_result']] = df_match[['home_team_goal', 'away_team_goal']].apply(getResultWinLoss, axis = 1, result_type='expand')



# Verify it is worked on the England Premier League input (as it has less NULLs)

df_match[df_match['league_name'] == 'England Premier League'][['home_team_result', 'away_team_result']].head()

# Check the column names, data types, and fields with null values

df_match.info()
# Drop the 'goal', card' and 'corner' columns

df_match.drop(columns=['goal', 'card', 'corner'], inplace=True)
# Change the data type of the 'date' column to Datetime



df_match['date'] = pd.to_datetime(df_match['date'])
# Verify if there are duplicate rows

sum(df_match.duplicated())
# Check the number of unique teams

print(df_match['home_team_id'].nunique())

print(df_match['away_team_id'].nunique())
# Check how many seasons are included

print('Number of seasons: {}'.format(df_match['season'].nunique()))

print('The seasons included are:')

df_match['season'].unique()
# Filter the data to keep only the matches from the 2015/2016 England Premier League



df_epl = df_match.query('league_name == "England Premier League" & season == "2015/2016"')

df_epl.head(3)
#How many teams participated in the 2015/2016 season

df_epl['home_team_id'].nunique()
# How many matches have been played in the '2015/2016' season

df_epl.shape
# Check if there are empty fields



df_epl.info()
df_epl[df_epl['home_player_2'].isnull()][['home_player_2', 'home_player_8', 'away_player_2']]
df_epl[df_epl['home_player_8'].isnull()][['home_player_2', 'home_player_8', 'away_player_2']]
df_epl[df_epl['away_player_2'].isnull()][['home_player_2', 'home_player_8', 'away_player_2']]
# Part 1 - for the home teams, count the number of goals scored and received



df_epl_goals_1 = df_epl.groupby('home_team_id', as_index=False)['home_team_goal', 'away_team_goal'].sum()

df_epl_goals_1.columns = ['team_id', 'goals_scored', 'goals_received']

df_epl_goals_1.head()
# Part 1 - for the home teams, count the number of matches won, lost and draws



df_epl_results_1 = df_epl.groupby(['home_team_id', 'home_team_result'], as_index=False)['match_id'].count()

df_epl_results_1.rename(columns={'home_team_id': 'team_id'}, inplace=True)

df_epl_results_1.head()
# Pivot the 'df_epl_results_1' so that each team appears on one row, and "win", "loss" and "draw" are the columns



df_epl_results_1 = pd.pivot_table(df_epl_results_1,  index = 'team_id', columns = ['home_team_result'], values = ['match_id'])

df_epl_results_1.columns = ['draw', 'loss', 'win']

df_epl_results_1.head()

# Before joining the 'results' table with the 'goals' table, I convert the 'home_team_id' index to a column



df_epl_results_1.reset_index(level=0, inplace=True)

df_epl_results_1.head()
# Merge the two Data Frames, so we have the full information in one Data Frame



df_epl_stats_1 = df_epl_results_1.merge(df_epl_goals_1, on='team_id', how='inner')

df_epl_stats_1.head()
# Part 2 - for the 'away' teams, count the number of goals scored and received



df_epl_goals_2 = df_epl.groupby('away_team_id', as_index=False)['away_team_goal', 'home_team_goal'].sum()

df_epl_goals_2.columns = ['team_id', 'goals_scored', 'goals_received']

df_epl_goals_2.head()
# Part 2 - for the 'away' teams, count the number of matches won, lost and draws



df_epl_results_2 = df_epl.groupby(['away_team_id', 'away_team_result'], as_index=False)['match_id'].count()

df_epl_results_2 = pd.pivot_table(df_epl_results_2,  index = 'away_team_id', columns = ['away_team_result'], values = ['match_id'])

df_epl_results_2.reset_index(level=0, inplace=True)

df_epl_results_2.columns = ['team_id', 'draw', 'loss', 'win']

df_epl_results_2.head()
# Merge the two Data Frames, so we have the full information in one Data Frame



df_epl_stats_2 = df_epl_results_2.merge(df_epl_goals_2, on='team_id', how='inner')

df_epl_stats_2.head()
# To get the real numbers of the '2015/2016' season, I need to sum up the two Data Frames

# In order to sum up the two Data Frames, I will make the 'team_id' as the Index



def makeColumnAsIndex(df, col):

    df.index = df[col]

    df.drop(columns = [col], inplace = True)

    return df

    

df_epl_stats_1 = makeColumnAsIndex(df_epl_stats_1, 'team_id')

df_epl_stats_2 = makeColumnAsIndex(df_epl_stats_2, 'team_id')

df_epl_stats_2.head()
# Sum up the two Data Frames



df_epl_stats_final = df_epl_stats_1 + df_epl_stats_2

df_epl_stats_final.head()
# Reset the index, so I can join the 'df_epl_stats_final' Data Frame to the 'df_team' Data Frame



df_epl_stats_final.reset_index(level=0, inplace=True)



# Join the 'df_match_stats_final' Data Frame to the 'df_team' Data Frame



df_epl_stats_final = df_team.merge(df_epl_stats_final, on = 'team_id', how='inner')

df_epl_stats_final.drop(columns=['team_id', 'team_short_name'], inplace=True)

df_epl_stats_final.head()
# Calculate the number of points for each team



def calculatePoints(results):

    return results['win'] * 3 + results['draw']



df_epl_stats_final['goals_diff'] = df_epl_stats_final['goals_scored'] - df_epl_stats_final['goals_received']

df_epl_stats_final['points'] = df_epl_stats_final.apply(calculatePoints, axis = 1)



# Sort teams by the number of points descending (and see the ranking)

df_epl_stats_final.sort_values(by="points", ascending=False, inplace=True)

df_epl_stats_final


Image("../input/imageseuropeanfootball/Football_results.png")
# Disply the column containing the scorers for each match

# The answer to Question 2 will be provided from the information in the 'scorers' column



df_epl['scorers']
# Display the number of total goals obtained from the stats table above

df_epl_stats_final['goals_scored'].sum()
# Verify if the number of 'official' goals (1026) match the number of goals from the 'scorers' columns 

# (it shouldn't match)



def getNumberOfGoals(items):

    if items is None or items == '':

        return 0

    items_list = items.split(';')

    return len(items_list)



# Add a column which stores the number of "goals" according to the 'scorers' column

df_epl['scorers_goalsNo'] = df_epl['scorers'].apply(getNumberOfGoals)



# Add a coumns which stores the official number of goals at the end of the match

df_epl['goalsNo'] = df_epl['home_team_goal'] + df_epl['away_team_goal']

df_epl.head()
# Display the matches for which the count doesn't match

df_epl[df_epl['scorers_goalsNo'] != df_epl['goalsNo']].head()
# Display the number of goals that don't match by comparing the columns 'goalsNo_from_scorers' and 'goalsNo'

# these are the 'own goals' - should be 35



sum(df_epl['goalsNo'] - df_epl['scorers_goalsNo'])
# Iterate through the values of the Series and extract the player IDs



def getCountsPerItem(columnItems):

    items_list = []



    for i, v in columnItems.items():

        if v is None or v == '':

            continue

            

        items = v.split(';')

        items_list.extend(items)

    

    # Create Pandas Series from list

    items_list = pd.Series(items_list)



    # Count the number of times an item appears

    items_count = items_list.value_counts()

    

    return items_count
scorers_list = getCountsPerItem(df_epl['scorers'])



# Print the top 10 scorers

top_scorers = scorers_list[:10]



for i, v in top_scorers.items():

    print(i,v)
# Display the total number of goals that are recorded in the 'scorer' column



sum(scorers_list.values)
# Transform Series to Data Frame



def seriesToFrame(series, column_name, index_name, indexToInt = True):

    dataFrame = series.to_frame(name=column_name)



    # Transform index into a column

    dataFrame.reset_index(level=0, inplace=True)



    # Rename index column 

    dataFrame.rename(columns={'index':index_name}, inplace=True)

    

    if indexToInt:

        # Change the data type of (index) column from string to int

        # only if indexToInt is True

        dataFrame[index_name] = dataFrame[index_name].astype(int)

    

    return dataFrame
top_scorers = seriesToFrame(top_scorers, 'goals', 'player_id')

top_scorers
# Join the 'top_scorers' Data Frame to the 'df_player' Data Frame to get the name of the players



top_scorers = df_player.merge(top_scorers, on="player_id", how="inner")

top_scorers.drop(columns=['player_id', 'birthday', 'height', 'weight'], inplace=True)

top_scorers.sort_values(by='goals', ascending=False, inplace=True)

top_scorers
# Display in a bar chart the top scorers



plt.subplots(figsize=(18,5))

plt.bar(top_scorers['player_name'], top_scorers['goals'])

plt.title('Top Scorers in the 2015/2016 England Premier League')

plt.xlabel('Players')

plt.ylabel('Number of goals')
Image("../input/imageseuropeanfootball/top_scorers.png")
# Understand how to retrieve the index of a given value from a Series

row = df_match.loc[4389] # return the row at index '4389' (returns a Series)

row[row == 1987033].index[0] # filter for the needed value and get its index 
# Drop the rows where the 'scorers' filed is null.

df_match_pos = df_match[df_match['scorers'].notna()]

df_match_pos.shape
# Verify if I can answer Question 3 based on 13045 matches



df_match_pos.loc[:, 'home_player_1_pos':'away_player_11_pos'].info()
def getPositionOfScorer(row):

    if row['scorers'] == None or row['scorers'] == '':

        return None

    

    positions = []

    

    scorers = row['scorers'].split(';')

    

    for scorer in scorers:

        player_id = float(scorer)

        

        # Find out in which column of the Data Frame this player ID appears

        # Example: home_player_3, away_player_10

        # It can be that the scorer doesn't appear in any column. This may be because he was changed during the match

        

        if row[row == player_id].size > 0:

            column = row[row == player_id].index[0]

            column_parts = column.split('_')

            #Create the name of the column that stores the player's position - created in the Data Exploration section

            position_column = column_parts[0] + '_player_' + column_parts[2] + '_pos'

            # Get the player's position - if the position is not available, then 'None' will be stored in 'positions'

            position = row[position_column]

            positions.append(position)

     

    positions = list(filter(None, positions))

    if len(positions) > 0:

        return ';'.join(positions)

    else:

        return None
df_match_pos['scorer_position'] = df_match_pos.apply(getPositionOfScorer, axis = 1)

df_match_pos.head()
df_match_pos[df_match_pos['scorer_position'].isnull()]
print(df_match_pos.loc[24520, 'scorers'])

df_match_pos.loc[24520, 'home_player_1':'away_player_11']
# Iterate through the values of the Series, extract the positions and count how many times each position appears



scorer_positions = getCountsPerItem(df_match_pos['scorer_position'])

scorer_positions
# The argument of this function is a Series. It calculates the proportions for the values of the Series

def getProportions(values):

    return values / sum(values)



proportions = getProportions(scorer_positions)

proportions
# Display in a bar chart the proportions of goals by player position



plt.subplots(figsize=(8,5))

plt.bar(proportions.index, proportions.values)

plt.title('Proportions of goals by player position')

plt.xlabel('Position')

plt.ylabel('Proportion of goals')
# See the dataframe columns used for answering Question 4

df_match[['home_team_goal', 'away_team_goal', 'corners_home_team', 'corners_away_team']].head()
# Remove the rows in which there is not value for the 'corners' columns



df_match_corners = df_match[df_match['corners_home_team'].notna() & df_match['corners_away_team'].notna()]

df_match_corners.shape
all_goals = df_match_corners['home_team_goal'].append(df_match_corners['away_team_goal'])

all_corners = df_match_corners['corners_home_team'].append(df_match_corners['corners_away_team'])
df_goal_corner = pd.DataFrame()

df_goal_corner['all_goals'] = all_goals

df_goal_corner['all_corners'] = all_corners

df_goal_corner['no'] = np.repeat(1, all_goals.size) 

df_goal_corner['all_corners'] = df_goal_corner['all_corners'].astype(int)

df_goal_corner.head()
df_goal_corner = df_goal_corner.groupby(['all_corners', 'all_goals']).sum()

df_goal_corner
df_goal_corner = df_goal_corner.pivot_table(index='all_corners', columns='all_goals', values='no')

df_goal_corner.fillna(0, inplace=True)



for col in df_goal_corner.columns:

    df_goal_corner[col] = df_goal_corner[col].astype(int)

    

df_goal_corner
# Get the values of the DataFrame in a 2D Numpy array

np_goal_corner = df_goal_corner.rename_axis('all_corners').values

np_goal_corner
# Plot a heatmap



fig, ax = plt.subplots(figsize=(14,14))

im = ax.imshow(df_goal_corner.rename_axis('all_corners').values)



# Show all ticks

ax.set_xticks(np.arange(len(df_goal_corner.columns)))

ax.set_yticks(np.arange(df_goal_corner.shape[0]))

# ... and label them with the respective list entries

ax.set_xticklabels(df_goal_corner.columns)

ax.set_yticklabels(list(df_goal_corner.index))

ax.set_ylabel("Number of corners")

ax.set_xlabel("Number of goals")



# Loop over data dimensions and create text annotations.

for i in range(df_goal_corner.shape[0]):

    for j in range(df_goal_corner.shape[1]):

        text = ax.text(j, i, np_goal_corner[i, j],

                       ha="center", va="center", color="w")



ax.set_title("Relationship between the number of corners and the number of goals")
# Check the relevant data

df_match[['cards_home_team', 'cards_away_team', 'home_team_result', 'away_team_result']].head()
# Remove the rows in which there is no value for the 'cards' columns



df_match_cards = df_match[df_match['cards_home_team'].notna() & df_match['cards_away_team'].notna()]

df_match_cards.shape
# Create two Series containg all cards and all results

all_cards = df_match_cards['cards_home_team'].append(df_match_cards['cards_away_team'])

all_results = df_match_cards['home_team_result'].append(df_match_cards['away_team_result'])



# Create a new Data Frame with two columns equal to the two Series created above

df_rel_card_result = pd.DataFrame()

df_rel_card_result['all_cards'] = all_cards

df_rel_card_result['all_results'] = all_results

df_rel_card_result['no'] = np.repeat(1, all_cards.size) #dummy column so I can perform the groupby count below

df_rel_card_result['all_cards'] = df_rel_card_result['all_cards'].astype(int)

df_rel_card_result.head()
# Group by the number of cards and the match outcome, and count the number of matches for each of this combination

df_rel_card_result.rename(columns={'no':'goalsNo'}, inplace=True)

df_rel_card_result = df_rel_card_result.groupby(['all_cards', 'all_results']).count()

df_rel_card_result
# Pivot the table above

df_rel_card_result = df_rel_card_result.pivot_table(index='all_cards', columns = 'all_results', values = 'goalsNo')



# Clean up the column names

df_rel_card_result.columns = ['draw', 'loss', 'win']



df_rel_card_result
# Replace the NaN values to 0

df_rel_card_result.fillna(0, inplace=True)



# Add column for the total number of matches for each possible number of cards received

df_rel_card_result['total'] = df_rel_card_result['draw'] + df_rel_card_result['loss'] + df_rel_card_result['win']



df_rel_card_result
ax = df_rel_card_result.loc[:,'draw':'win'].plot.bar(rot=0, figsize=(14,8))



plt.title('Relationship between the number of cards and the match outcome')

plt.xlabel('Number of cards')

plt.ylabel('Number of matches')
df_rel_card_result['perc_draw'] = df_rel_card_result['draw'] / df_rel_card_result['total'] * 100

df_rel_card_result['perc_loss'] = df_rel_card_result['loss'] / df_rel_card_result['total'] * 100

df_rel_card_result['perc_win'] = df_rel_card_result['win'] / df_rel_card_result['total'] * 100

df_rel_card_result
ax = df_rel_card_result.loc[:,'perc_draw':'perc_win'].plot.bar(rot=0, figsize=(14,8))



plt.title('Relationship between the number of cards and the match outcome')

plt.xlabel('Number of cards')

plt.ylabel('Percentage of matches')