import pandas as pd

import numpy as np

from scipy.stats import poisson

from glob import glob
# data ingestion - epl matches

file = '/kaggle/input/historical-matches-epl-championship/matches.csv'

matches = pd.read_csv(file, usecols=range(1,13))

matches.head()
# data processing

is_2020 = matches['Date'] > '2019-07'

last_10 = matches['Date'] > '2010-07'

#last_5 = (matches['Date'] > '2013-07') & (matches['Date'] < '2018-07')

last_5 = matches['Date'] > '2015-07'

last_1 = matches['Date'] > '2019-07'



# aggregate by home away team and return avg home away goals

matches[last_10].groupby(['HomeTeam', 'AwayTeam']).mean()
matches[last_10]['HomeTeam'].value_counts()
# head to head between every team pairing in the last 10 seasons

h2h = matches[last_5].groupby(['HomeTeam', 'AwayTeam']).mean()

matches[last_5].groupby(['HomeTeam', 'AwayTeam']).get_group(('Arsenal', 'Chelsea'))

#h2h.filter(like='Chelsea', axis=0)

#h2h.loc[('Arsenal', 'Chelsea')]

#h2h.loc[['Chelsea']]
# epl and championship teams home and away form last season

home_form = matches[last_1].groupby(['HomeTeam']).mean()

away_form = matches[last_1].groupby(['AwayTeam']).mean()

home_form.loc['Leeds'], away_form.loc['Leeds']
# modeling - get score function

home = 'Brighton'

away = 'Chelsea'



def get_score(home, away):

    # head to head results in last 5 seasons

    home_mean = h2h.loc[(home, away)][0]

    away_mean = h2h.loc[(home, away)][1]

    

    # simulate score by random sampling from parametrized Poisson distribution

    home_score = poisson.rvs(home_mean, size=1)[0]

    away_score = poisson.rvs(away_mean, size=1)[0]



    return (home_score, away_score)



get_score(home, away)
# simulate score over 10000 trials and plot histogram of most probable result

sims = {}

trials = 100

for i in range(trials):

    score = get_score(home, away)

    sims[score] = sims.get(score, 0) + 1



hist = []

for k, v in sims.items():

    p = v / trials

    hist.append((v, k, p))

    

hist.sort(reverse=True)

hist
# helper function returns match result (home win, away win, draw)

def get_result(home_score, away_score):

    if home_score > away_score:

        result = 'H'

    elif home_score < away_score:

        result = 'A'

    else:

        result = 'D'

    return result



get_result(2,1)
# modeling - get score function returns most probable result between two sides

home = 'Liverpool'

away = 'Leeds'

trials = 10000



def get_scores(home, away):

    # if head to head results exists

    try:

        # head to head results in last 5 seasons

        home_mean = h2h.loc[(home, away)][0]

        away_mean = h2h.loc[(home, away)][1]



    # if head to head results unavailable e.g. Leeds

    except KeyError:

        # home and away form last season

        home_scored = home_form.loc[home]['FTHG']

        home_conceded = home_form.loc[home]['FTAG']

        away_scored = away_form.loc[away]['FTAG']

        away_conceded = away_form.loc[away]['FTHG']

        

        # average over goals for and against each team

        home_mean = (home_scored + away_conceded) / 2

        away_mean = (away_scored + home_conceded) / 2



    # simulate score by random sampling from parametrized Poisson distribution

    home_scores = poisson.rvs(home_mean, size=trials, random_state=2).astype(str)

    away_scores = poisson.rvs(away_mean, size=trials, random_state=0).astype(str)

    

    # get most probable scoreline and outcome with associated probabilities

    scores = pd.DataFrame(data={'home':home_scores, 'away':away_scores})

    scores['result'] = scores['home'] + '-' + scores['away']

    score_predictions = scores['result'].value_counts()

    score_probability = round(score_predictions / trials * 100, 1)

    scores['outcome'] = scores.apply(lambda x: get_result(x.home, x.away), axis=1)

    outcome_predictions = scores['outcome'].value_counts()

    outcome_probability = round(outcome_predictions / trials * 100, 1)

  

    return score_predictions.index[0], score_probability[0], outcome_predictions.index[0], outcome_probability[0]

    

get_scores(home, away)
# data analysis - simulate 20/21 season opening week (week 1 games starting 2020-09-12)

home_teams = ['Fulham', 'Crystal Palace', 'Liverpool', 'West Ham', 'West Brom', 'Tottenham', 'Sheffield United', 'Brighton']

away_teams = ['Arsenal', 'Southampton', 'Leeds', 'Newcastle', 'Leicester', 'Everton', 'Wolves', 'Chelsea']



week1 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})

week1['Pred. Score'] = week1.apply(lambda x: get_scores(x.Home, x.Away)[0], axis=1)

week1['Probability'] = week1.apply(lambda x: get_scores(x.Home, x.Away)[1], axis=1).map('{:,.1f}%'.format)

week1['Pred. Result'] = week1.apply(lambda x: get_scores(x.Home, x.Away)[2], axis=1)

week1['Prob. (%)'] = week1.apply(lambda x: get_scores(x.Home, x.Away)[3], axis=1).map('{:,.1f}%'.format)

#week1
week1['Actual Score'] = ['0-3','1-0','4-3','0-2','0-3','0-1','0-2','1-3']

week1['Actual Result'] = ['A','H','H','A','A','A','A','A']

week1
# data analysis - simulate 20/21 season week 2 

home_teams = ['Everton', 'Leeds', 'Man United', 'Arsenal', 'Southampton', 'Newcastle', 'Chelsea', 'Leicester', 'Aston Villa', 'Wolves']

away_teams = ['West Brom', 'Fulham', 'Crystal Palace', 'West Ham', 'Tottenham', 'Brighton', 'Liverpool', 'Burnley', 'Sheffield United', 'Man City']



week2 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})

week2['Pred. Score'] = week2.apply(lambda x: get_scores(x.Home, x.Away)[0], axis=1)

week2['Probability'] = week2.apply(lambda x: get_scores(x.Home, x.Away)[1], axis=1).map('{:,.1f}%'.format)

week2['Pred. Result'] = week2.apply(lambda x: get_scores(x.Home, x.Away)[2], axis=1)

week2['Prob. (%)'] = week2.apply(lambda x: get_scores(x.Home, x.Away)[3], axis=1).map('{:,.1f}%'.format)

week2['Actual Score'] = ['5-2','4-3','1-3','2-1','2-5','0-3','0-2','4-2','1-0','1-3']

week2['Actual Result'] = ['H','H','A','H','A','A','A','H','H','A']

week2
# data analysis - simulate 20/21 season week 3 

home_teams = ['Brighton', 'Crystal Palace', 'West Brom', 'Burnley', 'Sheffield United', 'Tottenham', 'Man City', 'West Ham', 'Fulham', 'Liverpool']

away_teams = ['Man United', 'Everton', 'Chelsea', 'Southampton', 'Leeds', 'Newcastle', 'Leicester', 'Wolves', 'Aston Villa', 'Arsenal']



week3 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})

week3['Pred. Score'] = week3.apply(lambda x: get_scores(x.Home, x.Away)[0], axis=1)

week3['Probability'] = week3.apply(lambda x: get_scores(x.Home, x.Away)[1], axis=1).map('{:,.1f}%'.format)

week3['Pred. Result'] = week3.apply(lambda x: get_scores(x.Home, x.Away)[2], axis=1)

week3['Prob. (%)'] = week3.apply(lambda x: get_scores(x.Home, x.Away)[3], axis=1).map('{:,.1f}%'.format)

#week3['Actual Score'] = []

#week3['Actual Result'] = []

week3