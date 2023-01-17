import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import scipy



d_picks = pd.read_csv("/kaggle/input/csgo-professional-matches/picks.csv")
d_economy = pd.read_csv("/kaggle/input/csgo-professional-matches/economy.csv")
d_results = pd.read_csv("/kaggle/input/csgo-professional-matches/results.csv")
d_players = pd.read_csv("/kaggle/input/csgo-professional-matches/players.csv")
import asyncio
"""
ADAPTED FROM https://github.com/ryankirkman/pyglicko2/blob/master/glicko2.py

Copyright (c) 2009 Ryan Kirkman
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import math

class Player:
    # Class attribute
    # The system constant, which constrains
    # the change in volatility over time.
    _tau = 0.5

    def getRating(self):
        return (self.__rating * 173.7178) + 1500 

    def setRating(self, rating):
        self.__rating = (rating - 1500) / 173.7178

    rating = property(getRating, setRating)

    def getRd(self):
        return self.__rd * 173.7178

    def setRd(self, rd):
        self.__rd = rd / 173.7178

    rd = property(getRd, setRd)
     
    def __init__(self, rating = 1500, rd = 350, vol = 0.06):
        # For testing purposes, preload the values
        # assigned to an unrated player.
        self.setRating(rating)
        self.setRd(rd)
        self.vol = vol
        self.RDList = np.zeros(10)
        self.ratingList = np.zeros(10)
        self.outcomeList = np.zeros(10)
        self.games_played = 0
        
    def add_game_and_train(self, ratingEnemy, RDEnemy, outcome):
        self.ratingList[self.games_played] = ratingEnemy
        self.RDList[self.games_played] = RDEnemy
        self.outcomeList[self.games_played] = outcome
        
        self.games_played += 1
        
        if self.games_played >= 10:
            self.games_played = 0
            self.update_player(self.ratingList, self.RDList, self.outcomeList)
            
    def _preRatingRD(self):
        """ Calculates and updates the player's rating deviation for the
        beginning of a rating period.
        
        preRatingRD() -> None
        
        """
        self.__rd = math.sqrt(math.pow(self.__rd, 2) + math.pow(self.vol, 2))
        
    def update_player(self, rating_list, RD_list, outcome_list):
        """ Calculates the new rating and rating deviation of the player.
        
        update_player(list[int], list[int], list[bool]) -> None
        
        """
        # Convert the rating and rating deviation values for internal use.
        rating_list = [(x - 1500) / 173.7178 for x in rating_list]
        RD_list = [x / 173.7178 for x in RD_list]

        v = self._v(rating_list, RD_list)
        self.vol = self._newVol(rating_list, RD_list, outcome_list, v)
        self._preRatingRD()
        
        self.__rd = 1 / math.sqrt((1 / math.pow(self.__rd, 2)) + (1 / v))
        
        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * \
                       (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
        self.__rating += math.pow(self.__rd, 2) * tempSum
        
        
    def _newVol(self, rating_list, RD_list, outcome_list, v):
        """ Calculating the new volatility as per the Glicko2 system.
        
        _newVol(list, list, list) -> float
        
        """
        i = 0
        delta = self._delta(rating_list, RD_list, outcome_list, v)
        a = math.log(math.pow(self.vol, 2))
        tau = self._tau
        x0 = a
        x1 = 0
        
        while x0!=x1 and i<100:
            i+=1
            # New iteration, so x(i) becomes x(i-1)
            x0 = x1
            d = math.pow(self.__rating, 2) + v + math.exp(x0)
            h1 = -(x0 - a) / math.pow(tau, 2) - 0.5 * math.exp(x0) \
            / d + 0.5 * math.exp(x0) * math.pow(delta / d, 2)
            h2 = -1 / math.pow(tau, 2) - 0.5 * math.exp(x0) * \
            (math.pow(self.__rating, 2) + v) \
            / math.pow(d, 2) + 0.5 * math.pow(delta, 2) * math.exp(x0) \
            * (math.pow(self.__rating, 2) + v - math.exp(x0)) / math.pow(d, 3)
            x1 = x0 - (h1 / h2)

        return math.exp(x1 / 2)
        
    def _delta(self, rating_list, RD_list, outcome_list, v):
        """ The delta function of the Glicko2 system.
        
        _delta(list, list, list) -> float
        
        """
        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
        return v * tempSum
        
    def _v(self, rating_list, RD_list):
        """ The v function of the Glicko2 system.
        
        _v(list[int], list[int]) -> float
        
        """
        tempSum = 0
        for i in range(len(rating_list)):
            tempE = self._E(rating_list[i], RD_list[i])
            tempSum += math.pow(self._g(RD_list[i]), 2) * tempE * (1 - tempE)
        return 1 / tempSum
        
    def _E(self, p2rating, p2RD):
        """ The Glicko E function.
        
        _E(int) -> float
        
        """
        return 1 / (1 + math.exp(-1 * self._g(p2RD) * \
                                 (self.__rating - p2rating)))
        
    def _g(self, RD):
        """ The Glicko2 g(RD) function.
        
        _g() -> float
        
        """
        return 1 / math.sqrt(1 + 3 * math.pow(RD, 2) / math.pow(math.pi, 2))
        
    def did_not_compete(self):
        """ Applies Step 6 of the algorithm. Use this for
        players who did not compete in the rating period.
        did_not_compete() -> None
        
        """
        self._preRatingRD()
dataframes=[d_picks,d_economy,d_results]
for df in dataframes:
    print(df.describe())
    print(df.head())
d_results_old = d_results
d_results['date'] = pd.to_datetime(d_results['date'])
d_results = d_results[d_results['date']>pd.to_datetime(datetime.date(2016, 1, 1))]
d_results = d_results[d_results['match_id'].isin(d_players['match_id'])]

d_results.sort_values(by='date',ascending=True)
d_results['team_1'].value_counts()
d_players_old = d_players
d_players['date'] = pd.to_datetime(d_players['date'])
d_players = d_players[d_players['date']>pd.to_datetime(datetime.date(2016, 1, 1))]
d_players = d_players[d_players['match_id'].isin(d_results['match_id'])]

d_players.sort_values(by='date',ascending=False)
d_results['team_1'].value_counts().head(30).plot(kind='bar')
print(d_results['team_1'].value_counts()[:30])
best_teams = d_results['team_1'].value_counts()[:]

plt.show()
d_players['player_name'].value_counts().head(30).plot(kind='bar')
best_teams = best_teams.index.to_numpy()
best_players = d_players['player_name'].value_counts().index.to_numpy()
print(best_players)

d_results_bestteams = d_results_old[d_results_old['team_1'].isin(best_teams) & d_results_old['team_2'].isin(best_teams) & d_results_old['match_id'].isin(d_players['match_id'])]

date_split = datetime.date(2019, 1, 1)
d_results_bestteams_test = d_results_bestteams[d_results_bestteams['date'] >= pd.to_datetime(date_split)]
d_results_bestteams = d_results_bestteams[d_results_bestteams['date'] < pd.to_datetime(date_split)]

d_results_bestteams.sort_values(by='date',ascending=True)
index_dict = {}
players_dict = {} #Should remove this and just use hashing
glickoObjects = np.empty(len(best_teams), dtype=np.object)
glickoObjects_Players = np.empty(len(best_players), dtype=np.object)

for i, team in enumerate(best_teams):
    index_dict[team] = i
    glickoObjects[i] = Player()
    
for i, player in enumerate(best_players):
    players_dict[player] = i
    glickoObjects_Players[i] = Player()
games=0
upward_bias = 0.0

for _, row in d_results_bestteams.sort_values(by='date',ascending=True).iterrows():
    if games%1000==0:
        print(f"Now at game {games+1}")
    games+=1
#     print(i)
    team1 = row['team_1']
    team2 = row['team_2']
    
    d_players_match = d_players[d_players['match_id']==(row['match_id'])]
    players_team1 = d_players_match[d_players_match['team']==team1]['player_name'].unique()
    players_team2 = d_players_match[d_players_match['team']==team2]['player_name'].unique()
    
    Rdt1 = np.zeros(len(players_team1))
    Rating1 = np.zeros(len(players_team1))
    Rdt2 = np.zeros(len(players_team2))
    Rating2 = np.zeros(len(players_team2))
    
    for i, player in enumerate(players_team1):
        player_id = players_dict[player]
        Rdt1[i] = glickoObjects_Players[player_id].getRd()
        Rating1[i] = glickoObjects_Players[player_id].getRating()
    average_ratingt1 = np.average(Rating1, weights = 1/(Rdt1**2))
    sigmat1 = np.sqrt(np.sum(Rdt1**2))
        
    for i, player in enumerate(players_team2):
        player_id = players_dict[player]
        Rdt2[i] = glickoObjects_Players[player_id].getRd()
        Rating2[i] = glickoObjects_Players[player_id].getRating()
    average_ratingt2 = np.average(Rating2, weights = 1/(Rdt2**2))
    sigmat2 = np.sqrt(np.sum(Rdt2**2))
    
#     print(games)
    if row['map_winner']==1: #match_winner
        glickoObjects[index_dict[team1]].add_game_and_train(glickoObjects[index_dict[team2]].getRating(), glickoObjects[index_dict[team2]].getRd(), 1 + upward_bias)
        glickoObjects[index_dict[team2]].add_game_and_train(glickoObjects[index_dict[team1]].getRating(), glickoObjects[index_dict[team1]].getRd(), 0 + upward_bias)
        for player in players_team1:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt2, sigmat2, 1)
        for player in players_team2:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt1, sigmat1, 0)
            
    else:
        glickoObjects[index_dict[team1]].add_game_and_train(glickoObjects[index_dict[team2]].getRating(), glickoObjects[index_dict[team2]].getRd(), 0 + upward_bias)
        glickoObjects[index_dict[team2]].add_game_and_train(glickoObjects[index_dict[team1]].getRating(), glickoObjects[index_dict[team1]].getRd(), 1 + upward_bias)
        for player in players_team1:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt2, sigmat2, 0)
        for player in players_team2:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt1, sigmat1, 1)#ratingEnemy, RDEnemy, outcome
print(games)
def loss_t1_probability(rating1,RD1,rating2,RD2):
    func = lambda x: np.exp(-(x-rating1)**2/(2*RD1**2)) * scipy.special.erfc((x - rating2)/(RD2*np.sqrt(2)))/(2*np.sqrt(2*np.pi)*RD1)
    return scipy.integrate.quad(func,np.amin([rating1-20*RD1,rating2-20*RD2]),np.amax([rating1+20*RD1,rating2+20*RD2]),points = (rating1,rating2))[0]
print(loss_t1_probability(1500,350,1800,350))

def Kelly_crit(loss_prob, odds, fraction_money = 0.1):
    return ((1-loss_prob) - loss_prob/odds)*fraction_money
# print(best_teams)
team1 = 'fnatic'
team2 = 'Illuminar'
print(glickoObjects[index_dict[team1]].getRating())
print(glickoObjects[index_dict[team1]].getRd())
print(glickoObjects[index_dict[team1]].getRating() - 2*glickoObjects[index_dict[team1]].getRd())
print(glickoObjects[index_dict[team2]].getRating())
print(glickoObjects[index_dict[team2]].getRd())
print(glickoObjects[index_dict[team2]].getRating() - 2*glickoObjects[index_dict[team2]].getRd())

import random
profit_company = 0.05
variationality = 0.05

total_games=0
total_correctpredictions = 0
total_correctpredictions_probabilities = 0
total_correctpredictions_players = 0
total_correctpredictions_probabilities_players = 0
money = 1
money_array = np.zeros(len(d_results_bestteams_test['date']))
money_players = 1
money_players_array = np.zeros(len(d_results_bestteams_test['date']))

for _, row in d_results_bestteams_test.sort_values(by='date',ascending=True).iterrows():
    money_array[total_games] = money
    money_players_array[total_games] = money_players
    total_games += 1
    if total_games %1000==0:
        print(f"I am now doing  game {total_games}")
#     print(total_games)
    team1 = row['team_1']
    team2 = row['team_2']
    
    d_players_match = d_players[d_players['match_id']==(row['match_id'])]
    players_team1 = d_players_match[d_players_match['team']==team1]['player_name'].unique()
    players_team2 = d_players_match[d_players_match['team']==team2]['player_name'].unique()
    
    rating1 = glickoObjects[index_dict[team1]].getRating()
    rating2 = glickoObjects[index_dict[team2]].getRating()
    Rd1 = glickoObjects[index_dict[team1]].getRd()
    Rd2 = glickoObjects[index_dict[team2]].getRd()
    
    Rdt1 = np.zeros(len(players_team1))
    Rating1 = np.zeros(len(players_team1))
    Rdt2 = np.zeros(len(players_team2))
    Rating2 = np.zeros(len(players_team2))
    
    for i, player in enumerate(players_team1):
        player_id = players_dict[player]
        Rdt1[i] = glickoObjects_Players[player_id].getRd()
        Rating1[i] = glickoObjects_Players[player_id].getRating()
    average_ratingt1 = np.average(Rating1, weights = 1/(Rdt1**2))
    average_ratingt1 = np.sum(Rating1)
    sigmat1 = np.sqrt(np.sum(Rdt1**2))
        
    for i, player in enumerate(players_team2):
        player_id = players_dict[player]
        Rdt2[i] = glickoObjects_Players[player_id].getRd()
        Rating2[i] = glickoObjects_Players[player_id].getRating()
    average_ratingt2 = np.average(Rating2, weights = 1/(Rdt2**2))
    average_ratingt2 = np.sum(Rating2)
    sigmat2 = np.sqrt(np.sum(Rdt2**2))
    
    loss_prob_t1 = loss_t1_probability(rating1,Rd1,rating2,Rd2)
    loss_prob_t1_players = loss_t1_probability(average_ratingt1,sigmat1,average_ratingt2,sigmat2)
    
    temp_var = min(variationality,loss_prob_t1,loss_prob_t1_players,1-loss_prob_t1,1-loss_prob_t1_players)
    rand = random.uniform(-temp_var,temp_var)
    odds_team_1 = (loss_prob_t1+rand)/(1-loss_prob_t1-rand)*(1-profit_company)
    odds_team_2 = (1-loss_prob_t1-rand)/(loss_prob_t1+rand)*(1-profit_company)
    odds_team_1_players = (loss_prob_t1_players+rand)/(1-loss_prob_t1_players-rand)*(1-profit_company)
    odds_team_2_players = (1-loss_prob_t1_players-rand)/(loss_prob_t1_players+rand)*(1-profit_company)
    
#     print(row['map_winner'])
#     print(loss_prob_t1)
#     print(Kelly_crit(loss_prob_t1,odds))
#     print(money*(2 - row['map_winner'])*(odds*np.amax([0, Kelly_crit(loss_prob_t1,odds)]) - np.amax([0, Kelly_crit(1-loss_prob_t1,odds)])) + money*(row['map_winner'] - 1)*(odds * np.amax([0, Kelly_crit(1-loss_prob_t1,odds)]) - np.amax([0, Kelly_crit(loss_prob_t1,odds)])))
#     print(money)
#     print(money*(2 - row['map_winner'])*(odds*np.amax([0, Kelly_crit(loss_prob_t1,odds)])))
#     print(money*(row['map_winner'] - 1)*(odds * np.amax([0, Kelly_crit(1-loss_prob_t1,odds)]) - np.amax([0, Kelly_crit(loss_prob_t1,odds)])))
    if Rd1<50 and Rd2 <50:
        money += money*(2 - row['map_winner'])*(odds_team_1*np.amax([0, Kelly_crit(loss_prob_t1,odds_team_1)]) - np.amax([0, Kelly_crit(1-loss_prob_t1,odds_team_2)])) + money*(row['map_winner'] - 1)*(odds_team_2 * np.amax([0, Kelly_crit(1-loss_prob_t1,odds_team_2)]) - np.amax([0, Kelly_crit(loss_prob_t1,odds_team_1)]))
    if sigmat1 < 200 and sigmat2 < 200:
        money_players += money_players*(2 - row['map_winner'])*(odds_team_1_players*np.amax([0, Kelly_crit(loss_prob_t1_players,odds_team_1_players)]) - np.amax([0, Kelly_crit(1-loss_prob_t1_players,odds_team_2_players)])) + money_players*(row['map_winner'] - 1)*(odds_team_2_players * np.amax([0, Kelly_crit(1-loss_prob_t1_players,odds_team_2_players)]) - np.amax([0, Kelly_crit(loss_prob_t1_players,odds_team_1_players)]))
    
    if (rating1 - 2*Rd1 > rating2 - 2*Rd2) and row['map_winner']==1:
        total_correctpredictions +=1
    elif (rating1 - 2*Rd1 < rating2 - 2*Rd2) and row['map_winner']==2:
        total_correctpredictions +=1
#     if (rating1 > rating2) and row['map_winner']==1:
#         total_correctpredictions +=1
#     elif (rating1 < rating2) and row['map_winner']==2:
#         total_correctpredictions +=1
        
    if row['map_winner']==1:
        total_correctpredictions_probabilities += 1 - loss_prob_t1
    elif row['map_winner']==2:
        total_correctpredictions_probabilities += loss_prob_t1
        
    if (average_ratingt1 - 2*sigmat1 > average_ratingt2 - 2*sigmat2) and row['map_winner']==1:
        total_correctpredictions_players +=1
    elif (average_ratingt1 - 2*sigmat1 < average_ratingt2 - 2*sigmat2) and row['map_winner']==2:
        total_correctpredictions_players +=1
        
    if row['map_winner']==1:
        total_correctpredictions_probabilities_players += 1 - loss_prob_t1_players
    elif row['map_winner']==2:
        total_correctpredictions_probabilities_players += loss_prob_t1_players
        
    if row['map_winner']==1: #match_winner
        glickoObjects[index_dict[team1]].add_game_and_train(glickoObjects[index_dict[team2]].getRating(), glickoObjects[index_dict[team2]].getRd(), 1 + upward_bias)
        glickoObjects[index_dict[team2]].add_game_and_train(glickoObjects[index_dict[team1]].getRating(), glickoObjects[index_dict[team1]].getRd(), 0 + upward_bias)
        for player in players_team1:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt2, sigmat2, 1)
        for player in players_team2:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt1, sigmat1, 0)
    else:
        glickoObjects[index_dict[team1]].add_game_and_train(glickoObjects[index_dict[team2]].getRating(), glickoObjects[index_dict[team2]].getRd(), 0 + upward_bias)
        glickoObjects[index_dict[team2]].add_game_and_train(glickoObjects[index_dict[team1]].getRating(), glickoObjects[index_dict[team1]].getRd(), 1 + upward_bias)#ratingEnemy, RDEnemy, outcome
        for player in players_team1:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt2, sigmat2, 0)
        for player in players_team2:
            player_id = players_dict[player]
            glickoObjects_Players[player_id].add_game_and_train(average_ratingt1, sigmat1, 1)#ratingEnemy, RDEnemy, outcome

plt.plot(money_array)
plt.yscale('log',basey=2) 
plt.show()
plt.plot(money_players_array)
plt.yscale('log',basey=2) 
plt.show()

print(total_correctpredictions)
print(total_correctpredictions_players)
print(total_correctpredictions_probabilities)
print(total_correctpredictions_probabilities_players)
# print(total_games)
print(total_correctpredictions/total_games)
print(total_correctpredictions_players/total_games)
print(total_correctpredictions_probabilities/total_games)
print(total_correctpredictions_probabilities_players/total_games)
print(money)
print(money_players)
team1 = 'AGF'
team2 = 'BLUEJAYS'
print(glickoObjects[index_dict[team1]].getRating())
print(glickoObjects[index_dict[team1]].getRd())
print(glickoObjects[index_dict[team1]].getRating() - 2*glickoObjects[index_dict[team1]].getRd())
print(glickoObjects[index_dict[team2]].getRating())
print(glickoObjects[index_dict[team2]].getRd())
print(glickoObjects[index_dict[team2]].getRating() - 2*glickoObjects[index_dict[team2]].getRd())

loss_prob_t1 = loss_t1_probability(glickoObjects[index_dict[team1]].getRating(),glickoObjects[index_dict[team1]].getRd(),glickoObjects[index_dict[team2]].getRating(),glickoObjects[index_dict[team2]].getRd())
print(loss_prob_t1)
print(Kelly_crit(loss_prob_t1, 0.3))
print(Kelly_crit(1-loss_prob_t1, 2.3))
def match_prediction(team1, team2, odds1 = 0.87, odds2 = 0.87):
    rating1 = glickoObjects[index_dict[team1]].getRating()
    rating2 = glickoObjects[index_dict[team2]].getRating()
    Rd1 = glickoObjects[index_dict[team1]].getRd()
    Rd2 = glickoObjects[index_dict[team2]].getRd()
    loss_prob_t1 = loss_t1_probability(rating1,Rd1,rating2,Rd2)
    
    if rating1 - 2*Rd1 > rating2 - 2*Rd2:
        print(f"I think that {team1} is going to win with probability {1-loss_prob_t1}")
    else:
        print(f"I think that {team2} is going to win with probability {loss_prob_t1}")
    print(f"I would bet { Kelly_crit(loss_prob_t1,odds1)} on {team1}")
    print(f"I would bet { Kelly_crit(1 - loss_prob_t1,odds2)} on {team2}")

match_prediction('AGF','BLUEJAYS',0.3,2.3)
match_prediction('Nexus','Berzerk',0.28,2.5)
match_prediction('FATE','LDLC',0.556,1.24)

import math
print(*math.log(money_array))
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
    if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)                    
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

reduce_mem_usage(d_results)
reduce_mem_usage(d_economy)
reduce_mem_usage(d_players)
reduce_mem_usage(d_picks)
print(hash('ALEC')%100)
