import pandas as pd

import numpy as np

from scipy.stats import poisson

from glob import glob
'''

# data ingestion

files = glob('./E0*.csv')



matches = pd.DataFrame()

for file in files:

    #print(file)

    matches = pd.concat([matches, 

                         pd.read_csv(file, usecols=range(1,12), encoding = "latin", date_parser='pandas.to_datetime')]

                       ).drop_duplicates()





matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True)

matches = matches.sort_values('Date').reset_index(drop=True)

matches.dropna(axis=1, how='all', inplace=True)

matches.to_csv('./matches.csv')

matches.info()

matches.head()

'''



file = '/kaggle/input/historical-epl-match-results-19932020/matches.csv'

matches = pd.read_csv(file, usecols=range(1,13))

matches.head()
# data processing

is_2020 = matches['Date'] > '2019-07'

last_10 = matches['Date'] > '2010-07'

#last_5 = (matches['Date'] > '2013-07') & (matches['Date'] < '2018-07')

last_5 = matches['Date'] > '2015-07'



# aggregate by home away team and return avg home away goals

matches[last_10].groupby(['HomeTeam', 'AwayTeam']).mean()
matches[last_10]['HomeTeam'].value_counts()
# head to head between every team pairing in the last 10 seasons

h2h = matches[last_5].groupby(['HomeTeam', 'AwayTeam']).mean()

matches[last_5].groupby(['HomeTeam', 'AwayTeam']).get_group(('Arsenal', 'Chelsea'))

#h2h.filter(like='Chelsea', axis=0)

#h2h.loc[('Arsenal', 'Chelsea')]

#h2h.loc[['Chelsea']]
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
# modeling - get score function returns most probable result between two sides

home = 'Brighton'

away = 'Chelsea'

trials = 10000



def get_score(home, away):

    try:

        # head to head results in last 5 seasons

        home_mean = h2h.loc[(home, away)][0]

        away_mean = h2h.loc[(home, away)][1]



        # simulate score by random sampling from parametrized Poisson distribution

        home_scores = poisson.rvs(home_mean, size=trials).astype(str)

        away_scores = poisson.rvs(away_mean, size=trials).astype(str)



        scores = pd.DataFrame(data={'home':home_scores, 'away':away_scores})

        scores['result'] = scores['home'] + '-' + scores['away']

        predictions = scores['result'].value_counts()

        probability = round(predictions / trials * 100, 1)

        

        return predictions.index[0], probability[0]



    except KeyError:

        # return NA for teams with no head to head record in last 5 seasons        

        return 'N/A', 'N/A'

    

get_score(home, away)
# data analysis - simulate 20/21 season opening week (week 1 games starting 2020-09-12)

home_teams = ['Fulham', 'Crystal Palace', 'Liverpool', 'West Ham', 'West Brom', 'Tottenham', 'Sheffield United', 'Brighton']

away_teams = ['Arsenal', 'Southampton', 'Leeds', 'Newcastle', 'Leicester', 'Everton', 'Wolves', 'Chelsea']



week1 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})

week1['Predictions'] = week1.apply(lambda x: get_score(x.Home, x.Away)[0], axis=1)

week1['Probability%'] = week1.apply(lambda x: get_score(x.Home, x.Away)[1], axis=1)

week1