%matplotlib inline

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
df = pd.read_csv("../input/NFL Play by Play 2009-2017 (v4).csv")
df.head(10)
df.info()
print("Rows: ",len(df))
total_interceptions = len(df[df.InterceptionThrown == 1])
print("Total Count of Interceptions Thrown: ", total_interceptions)

total_fumbles = len(df[df.Fumble == 1])
print("Total Count of Fumbles: ", total_fumbles)

total_punts_blocked = len(df[df.PuntResult == 'Blocked'])
print("Total Count of Blocked Punts: ", total_punts_blocked)

total_fg_unsuccessful = len(df[(df.FieldGoalResult == 'Blocked') | (df.FieldGoalResult == 'No Good')])
print("Total Missed/Blocked Field Goals: ", total_fg_unsuccessful)

# because I'm not taking into account the hefty logic to determine turnover on downs -- "Giveaways"
total_giveaways = total_interceptions + total_fumbles + total_punts_blocked + total_fg_unsuccessful
print("Total Giveaways: ", total_giveaways)


# create a dict object containing the above calculations
giveaways = {'Interceptions': [total_interceptions],
             'Fumbles': [total_fumbles],
             'Blocked Punts': [total_punts_blocked],
             'Missed/Blocked FG': [total_fg_unsuccessful]}
giveaways_df = pd.DataFrame.from_dict(giveaways)

# plot the results in a simple bar chart
giveaways_df.plot(kind='bar')
# update data for consistent team names to just label franchises instead of cities to avoid outliers/ambiguity:

# LA --> LAR (avoid ambiguity with LAC)
df.loc[df.posteam == 'LA', 'posteam'] = 'LAR'
df.loc[df.DefensiveTeam == 'LA', 'DefensiveTeam'] = 'LAR'
df.loc[df.HomeTeam == 'LA', 'HomeTeam'] = 'LAR'
df.loc[df.AwayTeam == 'LA', 'AwayTeam'] = 'LAR'
df.loc[df.RecFumbTeam == 'LA', 'RecFumbTeam'] = 'LAR'
df.loc[df.PenalizedTeam == 'LA', 'PenalizedTeam'] = 'LAR'
df.loc[df.SideofField == 'LA', 'PenalizedTeam'] = 'LAR'

# STL --> LAR
df.loc[df.posteam == 'STL', 'posteam'] = 'LAR'
df.loc[df.DefensiveTeam == 'STL', 'DefensiveTeam'] = 'LAR'
df.loc[df.HomeTeam == 'STL', 'HomeTeam'] = 'LAR'
df.loc[df.AwayTeam == 'STL', 'AwayTeam'] = 'LAR'
df.loc[df.RecFumbTeam == 'STL', 'RecFumbTeam'] = 'LAR'
df.loc[df.PenalizedTeam == 'STL', 'PenalizedTeam'] = 'LAR'
df.loc[df.SideofField == 'STL', 'PenalizedTeam'] = 'LAR'

# SD --> LAC
df.loc[df.posteam == 'SD', 'posteam'] = 'LAC'
df.loc[df.DefensiveTeam == 'SD', 'DefensiveTeam'] = 'LAC'
df.loc[df.HomeTeam == 'SD', 'HomeTeam'] = 'LAC'
df.loc[df.AwayTeam == 'SD', 'AwayTeam'] = 'LAC'
df.loc[df.RecFumbTeam == 'SD', 'RecFumbTeam'] = 'LAC'
df.loc[df.PenalizedTeam == 'SD', 'PenalizedTeam'] = 'LAC'
df.loc[df.SideofField == 'SD', 'PenalizedTeam'] = 'LAC'

# JAC --> JAX
df.loc[df.posteam == 'JAC', 'posteam'] = 'JAX'
df.loc[df.DefensiveTeam == 'JAC', 'DefensiveTeam'] = 'JAX'
df.loc[df.HomeTeam == 'JAC', 'HomeTeam'] = 'JAX'
df.loc[df.AwayTeam == 'JAC', 'AwayTeam'] = 'JAX'
df.loc[df.RecFumbTeam == 'JAC', 'RecFumbTeam'] = 'JAX'
df.loc[df.PenalizedTeam == 'JAC', 'PenalizedTeam'] = 'JAX'
df.loc[df.SideofField == 'JAC', 'PenalizedTeam'] = 'JAX'
# update data to have an attribute for turnovers:
df['Turnover'] = df.apply(lambda x: 1 
                                 if ((x.InterceptionThrown == 1) | 
                                     (x.Fumble == 1) |
                                     (x.FieldGoalResult == 'Blocked') |
                                     (x.FieldGoalResult == 'No Good') |
                                     (x.PuntResult == 'Blocked'))
                                 else 0, axis=1)
# disable chained assignments --> for the logic I'm using this is simply an annoying warning and is populating correctly
pd.options.mode.chained_assignment = None 

# minimze the dataset for computational efficiency
df_scores = df[(df.qtr >= 4)]

# copy the columns(attributes) we declare in the list to a new dataframe to modify
results_attributes = ['HomeTeam','AwayTeam','posteam','PosTeamScore','DefensiveTeam','DefTeamScore','GameID','Date','qtr','PlayType','sp','Touchdown','FieldGoalResult','ExPointResult','TwoPointConv','Turnover','Safety','TimeSecs','Drive']
df_scores = df_scores[results_attributes]

# apply the lambda funstion to copy the PosTeamScores/DefTeamScores into HomeTeamScores and AwayTeam Scores
df_scores['HomeTeamScore'] = df_scores.apply(lambda x: x.PosTeamScore if x.HomeTeam == x.posteam else x.DefTeamScore, axis=1)
df_scores['AwayTeamScore'] = df_scores.apply(lambda x: x.PosTeamScore if x.AwayTeam == x.posteam else x.DefTeamScore, axis=1)

results_attributes = ['HomeTeam','HomeTeamScore','AwayTeam','AwayTeamScore','posteam','PosTeamScore','DefensiveTeam','DefTeamScore','GameID','Date','qtr','PlayType','sp','Touchdown','FieldGoalResult','ExPointResult','TwoPointConv','Turnover','Safety','TimeSecs','Drive']
df_scores = df_scores[results_attributes]
df_scores.head(20)
# get a list of the indices for the rows that indicate the End of Game
idx = df_scores[df_scores['PlayType'] == 'End of Game'].index.tolist()

# subtract 1 from the indices to use for accessing the row above the End of Game row
idx[:] = [x - 1 for x in idx]

# iterate over the list to access the values and copy them into the End of Game rows
for x in idx:
    home_score = df_scores.loc[x, 'HomeTeamScore']
    away_score = df_scores.loc[x, 'AwayTeamScore']
    y = x + 1
    if((df_scores.loc[y, 'PlayType'] == 'End of Game')):
        df_scores.loc[y, 'HomeTeamScore'] = home_score
        df_scores.loc[y, 'AwayTeamScore'] = away_score

# subset the dataframe to only include end of game results
Final_Results = df_scores[df_scores['PlayType'] == 'End of Game']
Final_Results.head(5)
# the GameID's that are not already in Final Results - because we already have the final scores for those games
df_scores = df_scores[~df_scores.GameID.isin(Final_Results.GameID)]

# Lets filter to only look at scoring plays first
df_sp = df_scores[df_scores.sp == 1]

#remove dups
df_sp = df_sp.drop_duplicates(subset=['GameID'], keep='last')

# function to add points to the Defensive Team's score whether they are Home or Away
def update_def_score(row):
    if (row['DefensiveTeam'] == row['HomeTeam']):
        if (row.Safety == 1): # There is a safety
            row.HomeTeamScore = row.HomeTeamScore+2
        else: row.HomeTeamScore = row.HomeTeamScore+6
    else:
        if (row.Safety == 1): # There is a safety
            row.AwayTeamScore = row.AwayTeamScore+2
        else: row.AwayTeamScore = row.AwayTeamScore+6
        return row

def update_score(row):
    if (row['posteam'] == row['AwayTeam']): #posteam is home team
        if (row.Touchdown == 1): # Touchdown ends game
            row.AwayTeamScore = row.AwayTeamScore+6
        elif ((row.PlayType == 'Field Goal')&(row.FieldGoalResult == 'Good')): # Field Goal to win game
            row.AwayTeamScore = row.AwayTeamScore+3
        elif ((row.PlayType == 'Extra Point')&(row.ExPointResult == 'made')): # Extra Point seals W
            row.AwayTeamScore = row.AwayTeamScore+1
        elif (row.TwoPointConv == 'Success'):
            row.AwayTeamScore = row.AwayTeamScore+2 # 2-pt conversion successful to win game
    elif(row['posteam'] == row['HomeTeam']): # posteam is away team
        if (row.Touchdown == 1): # Touchdown ends game
            row.HomeTeamScore = row.HomeTeamScore+6
        elif ((row.PlayType == 'Field Goal')&(row.FieldGoalResult == 'Good')): # Field Goal to win game
            row.HomeTeamScore = row.HomeTeamScore+3
        elif ((row.PlayType == 'Extra Point')&(row.ExPointResult == 'made')): # Extra Point seals W
            row.HomeTeamScore = row.HomeTeamScore+1
        elif (row.TwoPointConv == 'Success'):
            row.HomeTeamScore = row.HomeTeamScore+2 # 2-pt conversion successful to win game
    return row


# update the scores using apply(function)
d_sp = df_sp[((df_sp['Turnover'] == 1)&(df_sp['Touchdown'] == 1))&(df_sp['Safety'] == 1)].apply(update_def_score, axis = 1)
o_sp = df_sp[(df_sp['Turnover'] == 0) & (df_sp['Safety'] == 0)].apply(update_score, axis = 1)
d_sp = d_sp.append(o_sp)
# append to Final Results DF
Final_Results = Final_Results.append(d_sp)
# the GameID's that are not already in Final Results - because we already have the final scores for those games
df_scores = df_scores[~df_scores.GameID.isin(Final_Results.GameID)]

#remove dups
df_scores = df_scores.drop_duplicates(subset=['GameID'], keep='last')


# append the Non-Scoring endings to the Final Results
Final_Results = Final_Results.append(df_scores, sort = True)
# remove the duplicates
Final_Results = Final_Results.drop_duplicates(subset=['GameID'], keep='last')
print('Total Unique Games in the data:             ',len(df.GameID.unique()))
print('Total Games in the Final Results DF:        ',len(Final_Results))
print('Total Unique Games in the Final Results DF: ',len(Final_Results.GameID.unique()))



# drop the listed columns
not_needed = ['PosTeamScore', 'DefTeamScore','posteam','DefensiveTeam','qtr','PlayType','sp','Touchdown','FieldGoalResult','ExPointResult','TwoPointConv','Turnover','Safety','TimeSecs']
Final_Results = Final_Results.drop(columns=not_needed)

# winning team
Final_Results['WinningTeam'] = Final_Results.apply(lambda x: x.HomeTeam if x.HomeTeamScore > x.AwayTeamScore else x.AwayTeam, axis=1)

# point differential between the winning and losing teams
Final_Results['PointSpread'] = Final_Results['HomeTeamScore'] - Final_Results['AwayTeamScore']
Final_Results['PointSpread'] = Final_Results['PointSpread']

# taking a look:
Final_Results.head(5)
WinningTeam = Final_Results['WinningTeam'].value_counts()
ax = WinningTeam.plot.bar(figsize=(22,10),rot=0,)
ax.set_title("Each Team's Number of Games Won", fontsize=24,)
ax.set_xlabel("Team", fontsize=18)
ax.set_ylabel("# of Wins", fontsize=14)
ax.set_alpha(0.8)

# set individual bar lables using above list
for i in ax.patches:
    # get_x: width; get_height: verticle
    ax.text(i.get_x()+.02, i.get_height()+1, str(round((i.get_height()), 2)), fontsize=10, color='black',rotation=0)
result_df = Final_Results.groupby('HomeTeam')['PointSpread'].mean()
ax = result_df.plot.bar(figsize=(22,10),rot=0,color='orange', width=1)
ax.set_title("Average Point Differentials by Team", fontsize=24)
ax.set_xlabel("Team", fontsize=18)
ax.set_ylabel("Average Point Differential", fontsize=14)
ax.set_alpha(0.6)

