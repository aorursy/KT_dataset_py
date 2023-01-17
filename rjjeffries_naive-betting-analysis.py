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
def stake(odds):

    if 1/odds > 0.9: return 20

    if 1/odds > 0.8: return 10

    if 1/odds > 0.5: return 5

    if 1/odds > 0.2: return 2



matches['safest_bet_stake'] = matches.safest_odds.apply(stake)

matches['safest_bet_payout']= matches.safest_odds*matches.safest_bet_stake

matches.loc[~(matches.safest_outcome==matches.result),'safest_bet_payout']=0



matches['longshot_bet_stake'] = matches.longshot_odds.apply(stake)

matches['longshot_bet_payout']= matches.longshot_odds*matches.longshot_bet_stake

matches.loc[~(matches.longshot_outcome==matches.result),'longshot_bet_payout']=0



matches.head(10)
def bet(matches):

    percent_correct1=matches[~(matches.safest_bet_payout==0)].shape[0]/matches.shape[0]

    #print ("correct bets in case 1:")

    #print (percent_correct1)



    percent_correct2=matches[~(matches.longshot_bet_payout==0)].shape[0]/matches.shape[0]

    #print ("correct bets in case 2:")

    #print (percent_correct2)



    s_net_investment=matches.safest_bet_stake.sum()

    #print ("Safe investment:")

    #print (s_net_investment)



    l_net_investment=matches.longshot_bet_stake.sum()

    #print ("Long investment:")

    #print (l_net_investment)

    s_returns=sum(matches.safest_bet_payout)

    l_returns=sum(matches.longshot_bet_payout.fillna(0))



    #print ("Returns 1:")

    #print (s_returns)

    #print ("Returns 2:")

    #print (l_returns)

    

    loss1=(s_returns-s_net_investment)/s_net_investment

    loss2=(l_returns-l_net_investment)/l_net_investment



    print ("Net loss 1:")

    print (loss1*100)

    print ("Net loss 2:")

    print (loss2*100)



    

for l, df in matches.groupby('league_id'):

    print ("League %s" % leagues[leagues['id']==l]['name_y'])

    bet(df)

    print ( df[df.result=='D'].shape[0]/df.shape[0])

    print(20*'*')
#compute percentage loss

loss1=(s_returns-s_net_investment)/s_net_investment

loss2=(l_returns-l_net_investment)/l_net_investment



print ("Net loss 1:")

print (loss1)

print ("Net loss 2:")

print (loss2)
leagues[leagues['id']==1729]['name_y']