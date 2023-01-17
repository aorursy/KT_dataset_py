##import pandas module

import pandas as pd
##Read the file which is available in same code folder

matches=pd.read_csv(r'C:\Users\mgopalan\Music\Mani_notebook\matches.csv')

deliveries=pd.read_csv(r'C:\Users\mgopalan\Music\Mani_notebook\deliveries.csv')
#read the first 5 rows in both the dataframe
##We can do match analysis using matches dataframe##

matches.head()
##Analysis can be done on individual players##

deliveries.head()
##Search a condition in dataframe. We need to find the represenation of Kohli on dataframe

deliveries[deliveries.batsman.str.contains('Kohli')]

##we found batsman name is V Kohli
##Find total runs scored by Virat Kohli

deliveries[deliveries.batsman=='V Kohli'].batsman_runs.sum()
##Top 5 Run scorer

top_run_scorer=deliveries.groupby('batsman').sum().batsman_runs.sort_values(ascending=False).head()

print(top_run_scorer)
##Plot top run scorer

%matplotlib inline

top_run_scorer.plot(kind='bar')
#Strike rate of each Bowler

#1. Number of deliveries of each bowler

del_bowled=deliveries.groupby('bowler').bowler.count()

del_bowled.head()





#2.number of wickets taken

#check the unique wicket categories

deliveries.dismissal_kind.unique()
#create an array of valid dismisal kind

valid_dis_kind=['caught', 'bowled','lbw', 'caught and bowled',

                   'stumped','hit wicket']
#filter the data and pick only valid dsimisal kind

valid_dis=deliveries[deliveries.dismissal_kind.isin(valid_dis_kind)]
valid_dis.head()
wickets_taken=valid_dis.groupby('batsman').bowler.count()

valid_dis.groupby('batsman').bowler.count().head()
##Max runs conceded by a bowler during super over

super_over_df=deliveries[deliveries.is_super_over==1]

deliveries[deliveries.is_super_over==1].head()
#runs by super bowler

runs_in_super_over=super_over_df.groupby('bowler').total_runs.sum()

super_over_df.groupby('bowler').total_runs.sum().head()
#Find max runs by super bowler

runs_in_super_over.sort_values()
#plot the graph

runs_in_super_over.sort_values().plot(kind='bar')
##Tos won vs matches won

#Total number of teams who won match and Toss

print("won match and Toss   :",matches[matches['toss_winner']==matches['winner']].toss_winner.shape[0])

##Team names who won match and Toss

print("*"*60)

print("Team names who won match and Toss")

matches[matches['toss_winner']==matches['winner']].toss_winner
##In Season 2010 number of wins for each team

won_team_2010=matches[matches.season==2010].groupby('winner').winner.count()

won_team_2010.sort_values(ascending=False)

#Plot the winning team

won_team_2010.plot(kind='bar')
##man(player) of the match player

man_of_match=matches.groupby('player_of_match').player_of_match.count()

#sorting man_of match

print(man_of_match.sort_values(ascending=False))





##ploting first 5 man of match players

man_of_match.sort_values(ascending=False)[0:5].plot(kind='bar')

#Matches at chinnaswami stadium

matches_chinnaswamy=matches[matches.venue.str.contains('Chinna')]

matches_chinnaswamy.head(2)
#Matches at chinnaswami stadium , RCB were winners

matches_chinnaswamy[matches_chinnaswamy.winner=='Royal Challengers Bangalore'].groupby('winner').winner.count()

#RCB @home ground (Chinnaswamy stdium)

matches_team1=matches_chinnaswamy[matches_chinnaswamy.team1=='Royal Challengers Bangalore'].team1.count()

matches_team2=matches_chinnaswamy[matches_chinnaswamy.team2=='Royal Challengers Bangalore'].team2.count()

print('RCB played @chinnaswamy stadium :  -->',matches_team1+matches_team2)
##RCB @home ground (Chinnaswamy stdium)Alternate method

matches[((matches.team1=='Royal Challengers Bangalore') | (matches.team2=='Royal Challengers Bangalore')) & (matches.venue=='M Chinnaswamy Stadium')].shape[0]

deliveries[deliveries.batsman.str.contains('Vill')]
deliveries.head(1)
matches.head(1)
#Merge both the data set on match id and id



match_deliveries=pd.merge(matches,deliveries,left_on='id',right_on='match_id')
#Check the new dataframe

pd.set_option('display.max_columns',55)

match_deliveries.head(2)
#Each season, total runs :Palyer AB de Villers



run_AB=match_deliveries[match_deliveries.batsman=='AB de Villiers'].groupby('season').batsman_runs.sum()

run_AB
#plot a graph

run_AB.plot(kind='pie')