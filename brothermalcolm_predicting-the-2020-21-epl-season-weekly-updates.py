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
week3['Actual Score'] = ['2-3','1-2','3-3','0-1','0-1','1-1','2-5','4-0','0-3','3-1']
week3['Correct Scoreline'] = week3['Actual Score'] == week3['Pred. Score']
week3['Actual Result'] = week3['Actual Score'].apply(lambda x: get_result(x[0], x[-1]))
week3['Correct Result'] = week3['Actual Result'] == week3['Pred. Result']
week3
print('scoreline accuracy: %f' % (week3['Correct Scoreline'].sum() / week3['Correct Scoreline'].count()))
print('result accuracy: %f' % (week3['Correct Result'].sum() / week3['Correct Result'].count()))
# data analysis - simulate 20/21 season week 4
home_teams = ['Chelsea','Everton','Leeds','Newcastle','Leicester','Southampton','Arsenal','Wolves','Man United','Aston Villa']
away_teams = ['Crystal Palace','Brighton','Man City','Burnley','West Ham','West Brom','Sheffield United','Fulham','Tottenham','Liverpool']

week4 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})
week4['Pred. Score'] = week4.apply(lambda x: get_scores(x.Home, x.Away)[0], axis=1)
week4['Probability'] = week4.apply(lambda x: get_scores(x.Home, x.Away)[1], axis=1).map('{:,.1f}%'.format)
week4['Pred. Result'] = week4.apply(lambda x: get_scores(x.Home, x.Away)[2], axis=1)
week4['Prob. (%)'] = week4.apply(lambda x: get_scores(x.Home, x.Away)[3], axis=1).map('{:,.1f}%'.format)
week4['Actual Score'] = ['4-0','4-2','1-1','3-1','0-3','2-0','1-0','2-1','1-6','7-2']
week4['Correct Scoreline'] = week4['Actual Score'] == week4['Pred. Score']
week4['Actual Result'] = week4['Actual Score'].apply(lambda x: get_result(x[0], x[-1]))
week4['Correct Result'] = week4['Actual Result'] == week4['Pred. Result']
week4
print('scoreline accuracy: %f' % (week4['Correct Scoreline'].sum() / week4['Correct Scoreline'].count()))
print('result accuracy: %f' % (week4['Correct Result'].sum() / week4['Correct Result'].count()))
# data analysis - simulate 20/21 season week 5
home_teams = ['Everton','Chelsea','Man City','Newcastle','Sheffield United','Crystal Palace','Tottenham','Leicester','West Brom','Leeds']
away_teams = ['Liverpool','Southampton','Arsenal','Man United','Fulham','Brighton','West Ham','Aston Villa','Burnley','Wolves']

week5 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})
week5['Pred. Score'] = week5.apply(lambda x: get_scores(x.Home, x.Away)[0], axis=1)
week5['Probability'] = week5.apply(lambda x: get_scores(x.Home, x.Away)[1], axis=1).map('{:,.1f}%'.format)
week5['Pred. Result'] = week5.apply(lambda x: get_scores(x.Home, x.Away)[2], axis=1)
week5['Prob. (%)'] = week5.apply(lambda x: get_scores(x.Home, x.Away)[3], axis=1).map('{:,.1f}%'.format)
week5['Actual Score'] = ['2-2','3-3','1-0','1-4','1-1','1-1','3-3','0-1','0-0','0-1']
week5['Correct Scoreline'] = week5['Actual Score'] == week5['Pred. Score']
week5['Actual Result'] = week5['Actual Score'].apply(lambda x: get_result(x[0], x[-1]))
week5['Correct Result'] = week5['Actual Result'] == week5['Pred. Result']
week5
print('scoreline accuracy: %f' % (week5['Correct Scoreline'].sum() / week5['Correct Scoreline'].count()))
print('result accuracy: %f' % (week5['Correct Result'].sum() / week5['Correct Result'].count()))
# data analysis - simulate 20/21 season week 6
home_teams = ['Aston Villa','West Ham','Fulham','Man United','Liverpool','Southampton','Wolves','Arsenal','Brighton','Burnley']
away_teams = ['Leeds','Man City','Crystal Palace','Chelsea','Sheffield United','Everton','Newcastle','Leicester','West Brom','Tottenham']

week6 = pd.DataFrame(data={'Home':home_teams, 'Away':away_teams})
week6['Pred. Score'] = week6.apply(lambda x: get_scores(x.Home, x.Away)[0], axis=1)
week6['Probability'] = week6.apply(lambda x: get_scores(x.Home, x.Away)[1], axis=1).map('{:,.1f}%'.format)
week6['Pred. Result'] = week6.apply(lambda x: get_scores(x.Home, x.Away)[2], axis=1)
week6['Prob. (%)'] = week6.apply(lambda x: get_scores(x.Home, x.Away)[3], axis=1).map('{:,.1f}%'.format)
#week6['Actual Score'] = []
#week6['Correct Scoreline'] = week6['Actual Score'] == week6['Pred. Score']
#week6['Actual Result'] = week6['Actual Score'].apply(lambda x: get_result(x[0], x[-1]))
#week6['Correct Result'] = week6['Actual Result'] == week6['Pred. Result']
week6
