#include libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import sqlite3



#load the data. (P.S - copied this from Yoni Lev's Kernel)

with sqlite3.connect('../input/database.sqlite') as con:

    countries = pd.read_sql_query("SELECT * from Country", con)

    matches = pd.read_sql_query("SELECT * from Match", con)

    leagues = pd.read_sql_query("SELECT * from League", con)

    teams = pd.read_sql_query("SELECT * from Team", con)
#filter out & select only the relevant rows & columns

selected_countries = ['England','France','Germany','Italy','Spain','Netherlands','Portugal']

countries = countries[countries.name.isin(selected_countries)]

leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))



matches = matches[matches.league_id.isin(leagues.id)]

matches = matches[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','home_team_goal','away_team_goal','B365H', 'B365D' ,'B365A']]

matches.dropna(inplace=True)

matches.head()
#compute the match result (i.e Home win/Draw/Away win) from the goals data for the match

matches['result']='H'

matches.loc[matches.home_team_goal==matches.away_team_goal,"result"]='D'

matches.loc[matches.home_team_goal<matches.away_team_goal,"result"]='A'



#find the safest & the riskiest odds for each match

matches['safest_odds']=matches.apply(lambda x: min(x[11],x[12],x[13]),axis=1)

matches['longshot_odds']=matches.apply(lambda x: max(x[11],x[12],x[13]),axis=1)



#find the match outcome corresponding to the safest & riskiest odds

matches['safest_outcome']='H'

matches.loc[matches.B365D==matches.safest_odds,"safest_outcome"]='D'

matches.loc[matches.B365A==matches.safest_odds,"safest_outcome"]='A'



matches['longshot_outcome']='A'

matches.loc[matches.B365D==matches.longshot_odds,"longshot_outcome"]='D'

matches.loc[matches.B365H==matches.longshot_odds,"longshot_outcome"]='H'
matches['safest_bet_payout']=matches.safest_odds*10

matches.loc[~(matches.safest_outcome==matches.result),'safest_bet_payout']=0



matches['longshot_bet_payout']=matches.longshot_odds*10

matches.loc[~(matches.longshot_outcome==matches.result),'longshot_bet_payout']=0



matches.head(10)
percent_correct1=matches[~(matches.safest_bet_payout==0)].shape[0]/matches.shape[0]

print ("correct bets in case 1:")

print (percent_correct1)



percent_correct2=matches[~(matches.longshot_bet_payout==0)].shape[0]/matches.shape[0]

print ("correct bets in case 2:")

print (percent_correct2)



net_investment=10*matches.shape[0]

print ("Net investment:")

print (net_investment)



returns1=sum(matches.safest_bet_payout)

returns2=sum(matches.longshot_bet_payout)



print ("Returns 1:")

print (returns1)

print ("Returns 2:")

print (returns2)

#compute percentage loss

loss1=(returns1-net_investment)/net_investment

loss2=(returns2-net_investment)/net_investment



print ("Net loss 1:")

print (loss1)

print ("Net loss 2:")

print (loss2)
