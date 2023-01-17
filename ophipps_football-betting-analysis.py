import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
df = pd.read_csv('soccer_database_csv.csv')
# looking at the data to make sure it is clear and agrees with what I pulled from the SQL database
df.head()
df.describe() #getting an idea as to what the data looks like to see if we can make any predictions
df.shape #getting an idea as to how much data we are working with
df.info() #checking we dont have any missing values
df.hist(figsize=(15,12)); #plotting some histograms to get some first impressions
plt.hist(df.home_team_goal, alpha=0.5, label='Home Goals')
plt.hist(df.away_team_goal, alpha=0.5, label='Away Goals')
plt.legend(loc='upper right')
plt.title('Goals scored Home vs Away');
bins = np.linspace(0, 12.5, 15)

plt.hist(df.Bet365_Home, bins, alpha=0.5, label='B365 Home Odds')
plt.hist(df.Bet365_Away, bins, alpha=0.5, label='B365 Away Odds')
plt.legend(loc='upper right')
plt.title('Bet 365 Odds by Match Outcome');
df['total_goals'] = df['home_team_goal'] + df['away_team_goal']
# creating a total goals column as this may be useful later on
df.tail() #just to check that this has been added
conditions = [
    (df['home_team_goal'] == df['away_team_goal']),
    (df['home_team_goal'] > df['away_team_goal']),
    (df['home_team_goal'] < df['away_team_goal'])]
choices = ['D', 'H', 'A']
df['result'] = np.select(conditions, choices) 

#creates a new column that will give a letter to indicate result of match. H = Home team win, A = Away team win, D = Draw
df.head() # check that this has added succesfully
df.total_goals.sum()
df.groupby('result')['total_goals'].sum().plot.bar(title='Total Goals by Match Outcome');
df.groupby('result').total_goals.value_counts().unstack(0).plot.bar(title='Count of Match Outcome by Total Goals');
result_hw = df.result == 'H' # mask for home win
result_aw = df.result == 'A' # mask for away win
result_d = df.result == 'D' # mask for draw
hw_prc = df.result[result_hw].count() / len(df) # getting a percentage of games that the home team has won
aw_prc = df.result[result_aw].count() / len(df) # getting a percentage of games that the away team has won 
d_prc = df.result[result_d].count() / len(df) # getting a percentage of games that are drawn 
# this will plot a pie chart for us
labels = 'Home_Win', 'Away_Win', 'Draw'
fracs = [hw_prc, aw_prc, d_prc]
explode = (0.1,0,0)
plt.axis("equal")
plt.title('Percentage of Club wins per Scenario')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
league_pl = df.league_name == 'England Premier League' #mask to filter English teams
league_bbva = df.league_name == 'Spain LIGA BBVA' #mask to filter Spanish teams
df['home_bet_average'] = (df['Bet365_Home'] + df['Ladbrokes_Home'] + df['WilliamHill_Home']) / 3
df['away_bet_average'] = (df['Bet365_Away'] + df['Ladbrokes_Away'] + df['WilliamHill_Away']) / 3
df['draw_bet_average'] = (df['Bet365_Draw'] + df['Ladbrokes_Draw'] + df['WilliamHill_Draw']) / 3
# this will give us a value which shows the average 
home_av_odds = df.home_bet_average.sum() / len(df)
away_av_odds = df.away_bet_average.sum() / len(df)
draw_av_odds = df.draw_bet_average.sum() / len(df)
# pltting a bar chart to see the average odds for each scenario
plt.bar('Home', home_av_odds, label='Average Home Odds')
plt.bar('Away', away_av_odds, label='Average Away Odds')
plt.bar('Draw', draw_av_odds, label='Average Draw Odds')

plt.xlabel('Match Outcome')
plt.ylabel('Average Odds')
plt.title('Average Betting Odds by Result Type')
plt.legend();
# Looks for when the highest odds of a team at home equal what the Bet365 are giving 
b365H_max = df[["Bet365_Home", "Ladbrokes_Home", "WilliamHill_Home"]].max(axis=1) == df.Bet365_Home
b365H_max_count = np.count_nonzero(b365H_max)

# Looks for when the highest odds of a team at home equal what the Ladbrokes are giving 
ladbrokesH_max = df[["Bet365_Home", "Ladbrokes_Home", "WilliamHill_Home"]].max(axis=1) == df.Ladbrokes_Home
ladbrokesH_max_count = np.count_nonzero(ladbrokesH_max)

# Looks for when the highest odds of a team at home equal what the Willliam Hill are giving 
williamhillH_max = df[["Bet365_Home", "Ladbrokes_Home", "WilliamHill_Home"]].max(axis=1) == df.WilliamHill_Home
williamhillH_max_count = np.count_nonzero(williamhillH_max)


print(b365H_max_count)
print(ladbrokesH_max_count)
print(williamhillH_max_count) #here we are just checking that the results are sensible
# Looks for when the highest odds of a team playing away equal what the Bet365 are giving 
b365A_max = df[["Bet365_Away", "Ladbrokes_Away", "WilliamHill_Away"]].max(axis=1) == df.Bet365_Away
b365A_max_count = np.count_nonzero(b365A_max)

# Looks for when the highest odds of a team  playing away equal what the Ladbrokes are giving 
ladbrokesA_max = df[["Bet365_Away", "Ladbrokes_Away", "WilliamHill_Away"]].max(axis=1) == df.Ladbrokes_Away
ladbrokesA_max_count = np.count_nonzero(ladbrokesA_max)

# Looks for when the highest odds of a team  playing away equal what the Willliam Hill are giving 
williamhillA_max = df[["Bet365_Away", "Ladbrokes_Away", "WilliamHill_Away"]].max(axis=1) == df.WilliamHill_Away
williamhillA_max_count = np.count_nonzero(williamhillA_max)


print(b365A_max_count)
print(ladbrokesA_max_count)
print(williamhillA_max_count)
# Looks for when the highest odds of a team playing away equal what the Bet365 are giving 
b365D_max = df[["Bet365_Draw", "Ladbrokes_Draw", "WilliamHill_Draw"]].max(axis=1) == df.Bet365_Draw
b365D_max_count = np.count_nonzero(b365D_max)

# Looks for when the highest odds of a team  playing away equal what the Ladbrokes are giving 
ladbrokesD_max = df[["Bet365_Draw", "Ladbrokes_Draw", "WilliamHill_Draw"]].max(axis=1) == df.Ladbrokes_Draw
ladbrokesD_max_count = np.count_nonzero(ladbrokesD_max)

# Looks for when the highest odds of a team  playing away equal what the Willliam Hill are giving 
williamhillD_max = df[["Bet365_Draw", "Ladbrokes_Draw", "WilliamHill_Draw"]].max(axis=1) == df.WilliamHill_Draw
williamhillD_max_count = np.count_nonzero(williamhillD_max)


print(b365D_max_count)
print(ladbrokesD_max_count)
print(williamhillD_max_count)
home_win_odds_count = b365H_max_count + ladbrokesH_max_count + williamhillH_max_count #total count of odds values for home
away_win_odds_count = b365A_max_count + ladbrokesA_max_count + williamhillA_max_count #total count of odds values for away
draw_odds_count = b365D_max_count + ladbrokesD_max_count + williamhillD_max_count #total count of odds values for a draw

b365H_prc = b365H_max_count / home_win_odds_count #getting percentage of Bet 365 Home Wins against the total
ladbrokesH_prc = ladbrokesH_max_count / home_win_odds_count #getting percentage of Ladbrokes Home Wins against the total
williamhillH_prc = williamhillH_max_count / home_win_odds_count #getting percentage of William Hill Home Wins against the total

b365A_prc = b365A_max_count / away_win_odds_count #getting percentage of Bet 365 Away Wins against the total
ladbrokesA_prc = ladbrokesA_max_count / away_win_odds_count #getting percentage of Ladbrokes Away Wins against the total
williamhillA_prc = williamhillA_max_count / away_win_odds_count #getting percentage of William Hill Away Wins against the total

b365D_prc = b365D_max_count / draw_odds_count #getting percentage of Bet 365 Draws against the total
ladbrokesD_prc = ladbrokesD_max_count / draw_odds_count #getting percentage of Ladbrokes Draws against the total
williamhillD_prc = williamhillD_max_count / draw_odds_count #getting percentage of William Hill Draws against the total

labels = 'Bet365', 'Ladbrokes', 'William_Hill'
fracs = [b365H_prc, ladbrokesH_prc, williamhillH_prc]
explode = (0,0,0)
plt.axis("equal")
plt.title('Best odds for Home wins per bookmaker')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
labels = 'Bet365', 'Ladbrokes', 'William_Hill'
fracs = [b365A_prc, ladbrokesA_prc, williamhillA_prc]
explode = (0,0,0)
plt.axis("equal")
plt.title('Best odds for Away wins per bookmaker')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
labels = 'Bet365', 'Ladbrokes', 'William_Hill'
fracs = [b365D_prc, ladbrokesD_prc, williamhillD_prc]
explode = (0,0,0)
plt.axis("equal")
plt.title('Best odds for Draws per bookmaker')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
# this will create a column that gives us the column name which contains the longest odds
df['longest_odds'] = df[["Bet365_Home", "Bet365_Away", "Bet365_Draw"]].idxmax(axis=1)
# This will remove everything before the underscore e.g the Bet365 prefix
df.longest_odds = df.longest_odds.str.split('_').str[1]
# This only takes the letter of the column, meaning we can compare directly with the 'result' column
df.longest_odds = df.longest_odds.str[0]
# I will check that this works
df.head()
df['beating_bookies_team'] = df.result == df.longest_odds 
#this will flag as true if the longest_odds and results columns match, showing us its applicable to get the team name
df.head()
df['beating_bookies_team'] = np.where( ( (df['longest_odds'] == 'A') & (df['beating_bookies_team'] == True ) ) , df.away_team, df.beating_bookies_team)
df['beating_bookies_team'] = np.where( ( (df['longest_odds'] == 'H') & (df['beating_bookies_team'] == True ) ) , df.home_team, df.beating_bookies_team)
df['beating_bookies_team'] = np.where( ( (df['longest_odds'] == 'D') & (df['beating_bookies_team'] == True ) ) , 'Draw', df.beating_bookies_team)


#This code will look at the 'beating_bookies_team' column above, and reassign the value
#if the Away team had the longest odds it will change to the Away team name
#if the Home team had the longest odds it will change to teh Home team
#if a Draw had the longest odds it will change to read 'Draw'
#if the 'longest_odds' did not match 'result', it wll read the boolean "False"
df.tail(100) #checking that this works - using 100 rows to check all versions of the above
df['beating_bookies_team'].value_counts()
df2 = df[df.beating_bookies_team != False]
df2['beating_bookies_team'].value_counts().plot.bar(title='Which team beat the most?');
df3 = df2[df2.beating_bookies_team != 'Draw']
df3['beating_bookies_team'].value_counts().plot.bar(title='Which team beat the bookies the most?');
#the below will create a variable depending on a filter, then count how many values are in that variable
win_beat_count = len(df[(df['beating_bookies_team'] != 'Draw') & (df['beating_bookies_team'] != False)])
draw_beat_count = len(df[df['beating_bookies_team'] == 'Draw'])
not_beat_count = len(df[df['beating_bookies_team'] == False])

print(win_beat_count) #check we get an expected output
print(draw_beat_count) #check that we get an expected output, can compare to the data above
print(not_beat_count) #check that we get an expected output, can compare to the data above
print(win_beat_count + draw_beat_count + not_beat_count) #make sure they add up to the total, 760
#now we have checked them, lets make them into percentages
win_beat_prc = win_beat_count / len(df)
draw_beat_prc = draw_beat_count / len(df)
not_beat_prc = not_beat_count / len(df)
#creating a pie chart to easily see this data
labels = 'Draw', 'Match Won', 'Not beaten'
fracs = [draw_beat_prc, win_beat_prc, not_beat_prc]
explode = (0,0.1,0.1)
plt.axis("equal")
plt.title('Beat the bookies!\nPercentage of times Bet365 was beaten by match result')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
win_beat_count_home = len(df[(df['beating_bookies_team'] != 'Draw') & (df['beating_bookies_team'] != False) & (df['longest_odds'] == 'H')])
win_beat_count_away = len(df[(df['beating_bookies_team'] != 'Draw') & (df['beating_bookies_team'] != False) & (df['longest_odds'] == 'A')])
# the aboev will look at what how many times the games were won by away or home teams


print(win_beat_count_home)
print(win_beat_count_away) #just checking that these are sensible amounts
win_beat_home_prc = win_beat_count_home / (win_beat_count + draw_beat_count)
win_beat_away_prc = win_beat_count_away / (win_beat_count + draw_beat_count)
draw_beat_prc_win = draw_beat_count / (win_beat_count + draw_beat_count)
#these are to get some percentage amounts for a pie chart
#creating a pie chart to easily see the breakdown of bookie beating scenarios
labels = 'Draw', 'Match Won by Home Team', 'Match Won by Away Team'
fracs = [draw_beat_prc_win, win_beat_home_prc, win_beat_away_prc]
explode = (0.05,0.05,0.05)
plt.axis("equal")
plt.title('Beat the bookies!\nPercentage of bookie beating bets by match outcome')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
pl_beat_teams = len(df.beating_bookies_team[league_pl].value_counts())-2 
#count of number of teams that have beaten bookies from PL, minus two to remove count of Draw and False
bbva_beat_teams = len(df.beating_bookies_team[league_bbva].value_counts())-2 
#count of number of teams that have beaten bookies from BBVA, minus two to remove count of Draw and False
pl_beat_teams_prc = pl_beat_teams / (pl_beat_teams + bbva_beat_teams) #getting some percentages for count of PL teams
bbva_beat_teams_prc = bbva_beat_teams / (pl_beat_teams + bbva_beat_teams) #getting some percentages for count of BBVA teams
#creating a pie chart to easily see this data
labels = 'Premier League Team', 'BBVA Team'
fracs = [pl_beat_teams_prc, bbva_beat_teams_prc]
explode = (0.05,0.05)
plt.axis("equal")
plt.title('Beat the bookies!\nPercentage of teams beating the bookies by league')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
pl_beats_count = len(df[(df['beating_bookies_team'] != 'Draw') & (df['beating_bookies_team'] != False) & (df['league_name'] == 'England Premier League')])
bbva_beats_count = len(df[(df['beating_bookies_team'] != 'Draw') & (df['beating_bookies_team'] != False) & (df['league_name'] == 'Spain LIGA BBVA')])
#creating a varibale to give us the total number of bookie beating matches that are associated with league
print(pl_beats_count)
print(bbva_beats_count)
print(pl_beats_count + bbva_beats_count) 
#testing that this data is correct
pl_beats_prc = pl_beats_count / win_beat_count #creating some percentages for a  graph
bbva_beats_prc = bbva_beats_count / win_beat_count #creating some percentages for a graph
#creating a pie chart to easily see this data
labels = 'Premier League Match', 'BBVA Match'
fracs = [pl_beats_prc, bbva_beats_prc]
explode = (0.05,0.05)
plt.axis("equal")
plt.title('Beat the bookies!\nPercentage of matches beating the bookies by league')
plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True);
