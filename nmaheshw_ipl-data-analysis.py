# module loads

import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# create a function for labeling #

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')



# datasource address

BALL_BY_BALL = "../input/indian-premier-league-csv-dataset/Ball_by_Ball.csv"

MATCHES = "../input/indian-premier-league-csv-dataset/Match.csv"

PLAYER_MATCH = "../input/indian-premier-league-csv-dataset/Player_Match.csv"

PLAYER = "../input/indian-premier-league-csv-dataset/Player.csv"

SEASON = "../input/indian-premier-league-csv-dataset/Season.csv"

TEAMS = "../input/indian-premier-league-csv-dataset/Team.csv"
# data read

deliveries = pd.read_csv(BALL_BY_BALL);

matches = pd.read_csv(MATCHES);

player_match = pd.read_csv(PLAYER_MATCH);

player = pd.read_csv(PLAYER);

season = pd.read_csv(SEASON);

teams = pd.read_csv(TEAMS);
# print column names for all datasets

print("Deliveries Dataset")

print(deliveries.columns.values)

print('_'*80)

print("Matches Dataset")

print(matches.columns.values)

print('_'*80)

print("Player Match Dataset")

print(player_match.columns.values)

print('_'*80)

print("Player Dataset")

print(player.columns.values)

print('_'*80)

print("Season Dataset")

print(season.columns.values)

print('_'*80)

print("Teams Dataset")

print(teams.columns.values)

print('_'*80)
# preview data for datasets

print("Deliveries Dataset")

print(deliveries.head())

print('_'*80)

print("Matches Dataset")

print(matches.head())

print('_'*80)

print("Player Match Dataset")

print(player_match.head())

print('_'*80)

print("Player Dataset")

print(player.head())

print('_'*80)

print("Season Dataset")

print(season.head())

print('_'*80)

print("Teams Dataset")

print(teams.head())

print('_'*80)
# print dataset info

print("Deliveries Dataset")

print(deliveries.info())

print('_'*80)

print("Matches Dataset")

print(matches.info())

print('_'*80)

print("Player Match Dataset")

print(player_match.info())

print('_'*80)

print("Player Dataset")

print(player.info())

print('_'*80)

print("Season Dataset")

print(season.info())

print('_'*80)

print("Teams Dataset")

print(teams.info())

print('_'*80)
# data cleaning

deliveries["Team_Batting"] = pd.merge(deliveries, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Team_Batting_Id',right_on='Team_Id')["Team_Short_Code"]

deliveries["Team_Bowling"] = pd.merge(deliveries, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Team_Bowling_Id',right_on='Team_Id')["Team_Short_Code"]

deliveries["Striker"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Striker_Id',right_on='Player_Id')["Player_Name"]

deliveries["Non_Striker"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Non_Striker_Id',right_on='Player_Id')["Player_Name"]

deliveries["Bowler"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Bowler_Id',right_on='Player_Id')["Player_Name"]

deliveries["Bowler"] = pd.merge(deliveries, player[["Player_Id","Player_Name"]], how='left', left_on='Bowler_Id',right_on='Player_Id')["Player_Name"]



deliveries.drop(["Team_Batting_Id","Team_Bowling_Id","Player_dissimal_Id","Fielder_Id","Bowler_Id","Non_Striker_Id","Striker_Id"], axis=1, inplace=True)



print(deliveries.head())



matches["Team_Name"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Team_Name_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Opponent_Team"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Opponent_Team_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Toss_Winner"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Toss_Winner_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Match_Winner"] = pd.merge(matches, teams[["Team_Id","Team_Short_Code"]], how='left', left_on='Match_Winner_Id',right_on='Team_Id')["Team_Short_Code"]

matches["Man_Of_The_Match"] = pd.merge(matches, player[["Player_Id","Player_Name"]], how='left', left_on='Man_Of_The_Match_Id',right_on='Player_Id')["Player_Name"]

matches.drop(["Team_Name_Id","Opponent_Team_Id","Toss_Winner_Id","Match_Winner_Id","Man_Of_The_Match_Id","Second_Umpire_Id","First_Umpire_Id"], axis=1, inplace=True)

print(matches.head())
matches["type"]="pre-qualifier"

# marking match type for season 1 and season 2 - knockout stages

for year in range(1,3):

    final_match_index = matches[matches['Season_Id']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "semifinal-2")

    matches = matches.set_value(final_match_index-2, "type", "semifinal-1")

    

# marking knockout stage for season 3

for year in range(3,4):

    final_match_index = matches[matches['Season_Id']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "3rd-place")

    matches = matches.set_value(final_match_index-2, "type", "semifinal-2")

    matches = matches.set_value(final_match_index-3, "type", "semifinal-1")

    

# marking knowckout stage for rest of seasons

for year in range(4,10):

    final_match_index = matches[matches['Season_Id']==year][-1:].index.values[0]

    matches = matches.set_value(final_match_index, "type", "final")

    matches = matches.set_value(final_match_index-1, "type", "qualifier-2")

    matches = matches.set_value(final_match_index-2, "type", "eliminator")

    matches = matches.set_value(final_match_index-3, "type", "qualifier-1")
# Total type of matches

matches.groupby(["type"])["Match_Id"].count()
print((matches['Man_Of_The_Match'].value_counts()).idxmax(),' : has most man of the match awards')

print(((matches['Match_Winner']).value_counts()).idxmax(),': has the highest number of match wins')
# plot for match count in each season

sns.countplot(x='Season_Id', data=matches)

plt.show()
# Plot for Matches count at  every stadium

plt.figure(figsize=(12,6))

sns.countplot(y='Venue_Name', data=matches)

plt.yticks(rotation='horizontal')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='Match_Winner', data=matches)

plt.xticks(rotation='vertical')

plt.show()