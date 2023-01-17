# packages

import pandas as pd

import numpy as np

import random

import datetime

import matplotlib.pyplot as plt

import seaborn as sns
# load CSV file from Kaggle

path = "../input/"

spread_scores = pd.read_csv(path + "nfl-scores-and-betting-data/spreadspoke_scores.csv")
# look at the 2018 season to get a sense of the data

spread_scores[spread_scores.schedule_season == 2018]
# look at the teams df

teams = pd.read_csv(path + "nfl-scores-and-betting-data/nfl_teams.csv")

teams.head()
# select 1979 through 2018

spread_scores = spread_scores[(spread_scores.schedule_season >= 1979) & (spread_scores.schedule_season <= 2018)]

spread_scores
# add id fields for home and away teams

spread_scores = spread_scores.merge(teams[['team_name', 'team_id']], left_on = 'team_home', right_on = 'team_name').rename(columns = {'team_id': 'team_home_id'})

spread_scores = spread_scores.merge(teams[['team_name', 'team_id']], left_on = 'team_away', right_on = 'team_name').rename(columns = {'team_id': 'team_away_id'})

spread_scores.drop(columns = ['team_name_x', 'team_name_y'], inplace = True)

spread_scores
# the bettor class represents a bettor with a strategy

class bettor():

    def __init__(self, **kwargs):

        self.alive = True

        self.picked = []

        self.life_span = 0

        self.strategy = kwargs['strategy']  ## do I need to unpack this from **kwargs, or does this work?

        self.strategy0_top_n = kwargs['strategy0_top_n']

        self.strategy1_n = kwargs['strategy1_n']

        #  self.team = kwargs['team_list']  ##list of agents on your team - reserve - need to code

        

    def make_pick(self, season, week): # make a pick based on your strategy

        if self.strategy == 0: # code the strategy of choosing the randomly amoung the n best remaining odds 

            try: # work around becuase it returns an error on short seasons - need to qc more

                idx = random.choice(

                    spread_scores[

                        (spread_scores.schedule_season == season) & 

                        (spread_scores.schedule_week == week) & 

                        (~spread_scores.team_favorite_id.isin(self.picked))

                    ].nsmallest(

                        self.strategy0_top_n, columns = 'spread_favorite', keep = 'all'

                    ).index

                )

                pick = spread_scores.loc[idx,'team_favorite_id']

                

            except: #  end the betting if no dataframe is returned

                idx = 'none'

                pick = 'none'      

    

        if self.strategy == 1: # code the strategy of choosing the nth best remaining odds 



            try: # returns an error on short seasons - need to check if there have been missing weeks

                idx = random.choice(self.select_n(season, week, self.strategy1_n))

                pick = spread_scores.loc[idx,'team_favorite_id']

                

            except: #  end the betting if no dataframe is returned

                idx = 'none'

                pick = 'none' 

            

        self.picked.append(pick)    

        return pick, idx

   

    def select_n(self, season, week, n):  

        #helper function for strategy #1 since it is tricky to find the nth best with all the duplicated spreads 

        if n == 1:

            idx_list = spread_scores[

                (spread_scores.schedule_season == season) & 

                (spread_scores.schedule_week == week) & 

                (~spread_scores.team_favorite_id.isin(self.picked))

                ].nsmallest(

                    n, columns = 'spread_favorite', keep = 'all'

                ).index.tolist()

            return idx_list

        else:

            idx_n_list = spread_scores[

                (spread_scores.schedule_season == season) & 

                (spread_scores.schedule_week == week) & 

                (~spread_scores.team_favorite_id.isin(self.picked))

                ].nsmallest(

                    n, columns = 'spread_favorite', keep = 'all'

                ).index.tolist()



            idx_n_1_list = spread_scores[

                (spread_scores.schedule_season == season) & 

                (spread_scores.schedule_week == week) & 

                (~spread_scores.team_favorite_id.isin(self.picked))

                ].nsmallest(

                    n-1, columns = 'spread_favorite', keep = 'all'

                ).index.tolist()



            idx_list = [i for i in idx_n_list if i not in idx_n_1_list]



            if idx_list ==  []:

                return self.select_n(season, week, n-1)  #recursion is fun

            else:

                return idx_list



        

    def survive(self, pick, idx, season, week): #check to see if the bettor survived the week

        if pick == 'none':  # if no pick, then toggle to dead and update life_span

            self.life_span = int(week)-1

            self.alive = False

        else:

            # ID the winner

            if spread_scores.loc[idx,'score_home'] > spread_scores.loc[idx,'score_away']:

                winner = spread_scores.loc[idx,'team_home_id']

            elif spread_scores.loc[idx,'score_home'] < spread_scores.loc[idx,'score_away']:

                winner = spread_scores.loc[idx,'team_away_id']  

            else:

                winner = 'tie'

            

            

            # if the winner was picked, then pass and keep going, if the winner was not picked, then toggle to dead and update life_span.   

            # print('season',season,'week',week,'pick',pick,'winner',winner)

            if pick == winner:

                pass

            else:

                self.life_span = int(week)-1

                self.alive = False

            

    def run_season(self, season): 

        # runs a season for a bettor, but I want to move this outside of the bettor class to allow more flexibility 

        # (e.g., bettors acting as a team)

        self.alive = True

        self.picked = []

        self.life_span = 0

    

        for week in range(1,18):

            p, pi = self.make_pick(season, str(week))

            self.survive(p, pi, season, str(week))

            if self.alive == False:

                break

                

        return self.life_span

swill = bettor(strategy = 0, strategy0_top_n = 3, strategy1_n = 1)

swill.run_season(2016)
swill = bettor(strategy = 1, strategy0_top_n = 1, strategy1_n = 3)

swill.run_season(2016)
# a function to run 40 years of simulations

def tournaments(bettors, strategies):

    

    results = pd.DataFrame()

    results['strategy'] = strategies

    

    for season in range (1979, 2019):

        life_spans = []

        for bettor_obj in bettors:

            life_spans.append(bettor_obj.run_season(season))

        results[season] = life_spans    

        maximum = max(life_spans)

        results[str(season)+'_take'] = results[season]==maximum

        results[str(season)+'_take'] = results[str(season)+'_take']/sum(results[str(season)+'_take'])        

        print('completed',season,'season')

    return results
# make a list of bettors and strategies for the tournament

bettor_list = []

strategy_list = []

bettors = 4

for n in range(1,10):

    for s in range(2):

        for b in range(bettors):

            swill = bettor(strategy = s, strategy0_top_n = n, strategy1_n = n)

            bettor_list.append(swill)

            strategy_list.append('s'+str(s)+'_'+'n'+str(n))



# print(bettor_list,strategy_list)
results = tournaments(bettor_list, strategy_list)
# crunch the results of the simulation

cols = results.columns.tolist()

take_cols = list(c for c in cols if 'take' in str(c))

results.groupby(by = 'strategy').sum()[take_cols]
pd.DataFrame(results.groupby(by = 'strategy').sum()[take_cols].sum(axis=1))