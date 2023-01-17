import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # more plotting

from sklearn.preprocessing import LabelEncoder

import math

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
#default settings for elo

mean_elo = 1500.

elo_width = 400.

k_factor = 64.

n_samples = 8000 #used for predicting
''' helper to classify each game as home win, away win or tie '''

def defineWinner(row):

    if row['fthg'] > row['ftag']:

        row['result'] = 1#'Home win'

    elif row['ftag'] > row['fthg']:

        row['result'] = 0#'Away win'

    elif row['fthg'] == row['ftag']:

        row['result'] = 0.5#'Tie'

    else: # For when scores are missing, etc (should be none)

        row['result'] = None

    return row



ginf = pd.read_csv('../input/ginf.csv', index_col=0)



ginf = ginf.apply(defineWinner, axis=1)
print(ginf.groupby('result')['result'].count())

years   = ginf.season.unique()

seasons = ginf.country.unique()
le = LabelEncoder()

le.fit(np.unique(np.concatenate((ginf['ht'].tolist(), ginf['at'].tolist()),axis=0)))



ginf['ht'] = le.transform(ginf.ht)

ginf['at'] = le.transform(ginf['at'])
ginf['ht_elo_before_game'] = 0

ginf['ht_elo_after_game']  = 0

ginf['at_elo_before_game'] = 0

ginf['at_elo_after_game']  = 0

n_teams = len(le.classes_)

print(n_teams)
def expected_score(rating_a, rating_b):

    """Returns the expected score for a game between the specified players

	http://footballdatabase.com/methodology.php

    """

    W_e = 1.0/(1+10**((rating_b - rating_a - 100)/elo_width))

    return W_e





def get_k_factor(rating, goals=0):

    """Returns the k-factor for updating Elo.

	http://footballdatabase.com/methodology.php

    """

    if not goals or goals == 1:

    	return k_factor



    if goals == 2:

    	return k_factor*1.5



    return k_factor*((11+goals)/8)



def calculate_new_elos(rating_a, rating_b, score_a, goals):

    """Calculates and returns the new Elo ratings for two players.

    score_a is 1 for a win by player A, 0 for a loss by player A, or 0.5 for a draw.

    """



    e_a = expected_score(rating_a, rating_b)

    e_b = 1. - e_a

    if goals > 0:

    	a_k = get_k_factor(rating_a, goals)

    	b_k = get_k_factor(rating_b)

    else:

    	a_k = get_k_factor(rating_a)

    	b_k = get_k_factor(rating_b, goals)



    new_rating_a = rating_a + a_k * (score_a - e_a)

    score_b = 1. - score_a

    new_rating_b = rating_b + b_k * (score_b - e_b)

    return new_rating_a, new_rating_b



def update_end_of_season(elos):

    """Regression towards the mean

    

    Following 538 nfl methods

    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/

    """

    #elos *= .75

    #elos += (.25*1505)

    diff_from_mean = elos - mean_elo

    elos -= diff_from_mean/3.

    return elos



def getWinner(row):

	epsilon = 1e-15

	if row['fthg'] > row['ftag']: #Home Win

		return (row['ht'], row['at'], 1-epsilon)

	elif row['ftag'] > row['fthg']: #Away Win

		return (row['ht'], row['at'], epsilon)

	elif row['fthg'] == row['ftag']: #Tie

		return (row['ht'], row['at'], 0.5)

	#else

	#	return (None, None)
print("Training...")

start=2012

end=2017

elo_per_season = {}

current_elos   = np.ones(shape=(n_teams)) * mean_elo



for year in range(start, end + 1):

	current_season = year

	games          = ginf[ginf['season']==year]



	for idx, game in games.iterrows():

		(ht_id, at_id, score) = getWinner(game)

        #update elo score

		ht_elo_before = current_elos[ht_id]

		at_elo_before = current_elos[at_id]

		ht_elo_after, at_elo_after = calculate_new_elos(ht_elo_before, at_elo_before, score, game['fthg']-game['ftag'])

		

		# Save updated elos

		ginf.at[idx, 'ht_elo_before_game'] = ht_elo_before

		ginf.at[idx, 'at_elo_before_game'] = at_elo_before

		ginf.at[idx, 'ht_elo_after_game'] = ht_elo_after

		ginf.at[idx, 'at_elo_after_game'] = at_elo_after



		#print "Score: ", game.result, "Goals:", "Predicted:", expected_score(ht_elo_before, at_elo_before), expected_score(at_elo_before, ht_elo_before), game['fthg']-game['ftag'], "Home Before:", ht_elo_before, " and After:", ht_elo_after, "Away Before:", at_elo_before, " and After:", at_elo_after



		current_elos[ht_id] = ht_elo_after

		current_elos[at_id] = at_elo_after



	elo_per_season[year] = current_elos.copy()

	current_elos         = update_end_of_season(current_elos)

ginf.head()
for year in range(2012, 2018):

    s = elo_per_season[year]

    print(year, "mean:", s.mean() , "min:", s.min(), "max:", s.max())
print("Predicting...")

start = 2013 #2012 is "the learning" year

samples       = ginf[ginf.season >= start].sample(n_samples)

samples       = ginf[ginf.season >= start].sample(n_samples)

loss          = 0

expected_list = []

epsilon       = 1e-15



y_true    = []

y_predicted = []

for row in samples.itertuples():

	ht_elo      = row.ht_elo_before_game

	at_elo      = row.at_elo_before_game

	w_expected = expected_score(ht_elo, at_elo)

	if w_expected >= .7:

		predicted = 1

	elif .4 <= w_expected and w_expected < .7:

		predicted = 2

	elif w_expected <= .4:

		predicted = 0



	y_true.append(row.result if row.result != .5 else 2)

	y_predicted.append(predicted)



	#print"Winner:",row.result, "Predicted:", predicted,  "Home: ", round(w_expected, 2), "(", ht_elo, ")", "Tie:", round(abs(.5-w_expected), 2), "(", at_elo, ")","Away: ", round(1-w_expected, 2)



	l = row.result - w_expected

	l = w_expected

	'''if l < epsilon:

		l = epsilon

	if l > 1-epsilon:

		l = 1-epsilon

	'''

	loss       += np.log(l)

	expected_list.append(l)



print("Loss:", loss/n_samples)

matrix = pd.DataFrame(confusion_matrix(y_true, y_predicted))

print("Matrix Code: ")

print(matrix)

print(classification_report(y_true, y_predicted, target_names=['away', 'home', 'tie']))

sns.distplot(expected_list, kde=False, bins=20)

plt.xlabel('Elo Expected Wins for Actual Winner')

plt.ylabel('Counts')

plt.show()

expected_list[:10]
def show_team_over_time(elo_per_season, team_id):

	x = []

	y = range(2012, 2017)

	

	for i in range(2012, 2017):

		x.append(elo_per_season[i][team_id])

	

	y_pos = np.arange(len(y)) 

	plt.bar(y_pos, x, align='center', alpha=0.5)

	plt.xticks(y_pos, y)

	plt.xlabel('Year')

	plt.ylabel('Elo Score Over Time')

	plt.show()
show_team_over_time(elo_per_season, le.transform(['Arsenal'])[0])