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
#compute the payout for betting Home win/Draw/Away win.

matches['home_bet_payout']=0

matches.loc[matches.result=='H','home_bet_payout']=matches.B365H



matches['draw_bet_payout']=0

matches.loc[matches.result=='D','draw_bet_payout']=matches.B365D



matches['away_bet_payout']=0

matches.loc[matches.result=='A','away_bet_payout']=matches.B365A



matches.head()
percent_correct_home=matches[~(matches.home_bet_payout==0)].shape[0]/matches.shape[0]

print ('correct bets in case we bet "Home win":')

print (percent_correct_home)



percent_correct_draw=matches[~(matches.draw_bet_payout==0)].shape[0]/matches.shape[0]

print ('correct bets in case we bet "Draw":')

print (percent_correct_draw)



percent_correct_away=matches[~(matches.away_bet_payout==0)].shape[0]/matches.shape[0]

print ('correct bets in case we bet "Away win":')

print (percent_correct_away)



assert(percent_correct_home+percent_correct_away+percent_correct_draw==1.0)
num_matches = matches.shape[0]



home_gain = sum(matches.home_bet_payout)

draw_gain = sum(matches.draw_bet_payout)

away_gain = sum(matches.away_bet_payout)



print ('we bet 1 euro on each one of the following matches')

print (num_matches)



print ('betting always 1 euro on "Home win" will get you')

print(home_gain)



print ('betting always 1 euro on "Draw" will get you')

print(draw_gain)



print ('betting always 1 euro on "Away win" will get you')

print(away_gain)
#compute percentage loss

home_loss = (home_gain-num_matches)/num_matches

draw_loss = (draw_gain-num_matches)/num_matches

away_loss = (away_gain-num_matches)/num_matches



print ('Net loss betting always "Home win":')

print (home_loss)

print ('Net loss betting always "Draw":')

print (draw_loss)

print ('Net loss betting always "Away win":')

print (away_loss)
#work in progress - not sure it is correct to compute mean on away_bet_payout for instance

matches_by_league_season = matches.groupby(('season','league_id'))



home_bet_means = matches_by_league_season.home_bet_payout.mean()

home_bet_means = home_bet_means.reset_index().pivot(index='season', columns='league_id', values='home_bet_payout')

home_bet_means.columns = [leagues[leagues.id==x].name.values[0] for x in home_bet_means.columns]

home_bet_means.index.name = 'Always betting "home win" ---------- Season'



home_bet_means.head(10)
draw_bet_means = matches_by_league_season.draw_bet_payout.mean()

draw_bet_means = draw_bet_means.reset_index().pivot(index='season', columns='league_id', values='draw_bet_payout')

draw_bet_means.columns = [leagues[leagues.id==x].name.values[0] for x in draw_bet_means.columns]

draw_bet_means.index.name = 'Always betting "draw" ---------- Season'

draw_bet_means.head(10)
away_bet_means = matches_by_league_season.away_bet_payout.mean()

away_bet_means = away_bet_means.reset_index().pivot(index='season', columns='league_id', values='away_bet_payout')

away_bet_means.columns = [leagues[leagues.id==x].name.values[0] for x in away_bet_means.columns]

away_bet_means.index.name = 'Always betting "away win" \n ----------\n Season'

away_bet_means.head(10)