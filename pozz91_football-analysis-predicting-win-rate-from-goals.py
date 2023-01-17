import pandas as pd

import numpy as np



match= pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')



#put date in correct format and set it as the index

date= pd.to_datetime(match.date.values)

match['date']=date

match.set_index('date', inplace=True)





#get the coloumn of results (wins, ties and losses)

win= np.where(match.home_score > match.away_score, 'win', None)

tie=np.where(match.home_score == match.away_score, 'tie', None)

loss= np.where(match.home_score < match.away_score, 'loss', None)



results=pd.DataFrame([win, tie, loss]).T

results

x=[value[value != None]  for value in results.values]

#x=np.array(x)

#x=x.tolist()

match['result']= x

match['result']=np.squeeze(match.result.tolist())



#get the number of goals

match['goals']= match.home_score + match.away_score





#home

home_teams=match.groupby(['home_team','result']).count()['city'].sort_values(ascending=False).reset_index().rename(columns={'city': 'count'})





home_matches=[]

for team in home_teams.home_team:

    tot_matches= home_teams[home_teams.home_team== team]['count'].sum()

    home_matches.append(tot_matches)

   

home_teams['home_matches']=home_matches

home_teams['pct_home_victory']= home_teams['count']/ home_teams['home_matches']





#away

away_teams=match.groupby(['away_team','result']).count()['city'].sort_values(ascending=False).reset_index().rename(columns={'city': 'count'})

away_teams.replace({'loss': 'win', 'win':'loss'}, inplace=True) #loss means victory for the away team



away_tot_matches=[]

for team in away_teams.away_team:

    tot_matches= away_teams[away_teams.away_team == team]['count'].sum()

    away_tot_matches.append(tot_matches)



away_teams['away_matches']= away_tot_matches

away_teams['pct_victory_away'] = away_teams['count']/away_teams['away_matches']





#adjusting terminology and index

home_teams.rename(columns={'result': 'home_results', 'count': 'home_count'}, inplace=True)

home_teams.set_index('home_team', inplace=True)

away_teams.rename(columns={'result': 'away_results', 'count': 'away_count'}, inplace=True)

away_teams.set_index('away_team', inplace=True)





#defining winners and loosers

home_winners= home_teams[home_teams.home_results=='win']

away_winners= away_teams[away_teams.away_results=='win']

home_losers= home_teams[home_teams.home_results=='loss']

away_losers= away_teams[away_teams.away_results=='loss']





#merging datasets

winners=pd.merge(home_winners, away_winners, left_index=True, right_index=True, how='inner')

losers=pd.merge(home_losers, away_losers, left_index=True, right_index=True, how='inner')

losers.rename(columns={'pct_home_victory': 'pct_home_defeats', 'pct_victory_away': 'pct_away_defeats'}, inplace=True)



winners['tot_count']= winners.home_count + winners.away_count

winners['tot_matches']= winners.home_matches + winners.away_matches

winners['tot_pct_victory']= winners.tot_count/winners.tot_matches

winners= winners[winners.tot_matches >= 100] #getting only clubs who have played at least 100 matches

winners_pct= winners[['pct_home_victory', 'pct_victory_away', 'tot_pct_victory']]



losers['tot_count']= losers.home_count + losers.away_count

losers['tot_matches']= losers.home_matches + losers.away_matches

losers['tot_pct_defeats']= losers.tot_count/losers.tot_matches

losers= losers[losers.tot_matches >= 100] #getting only clubs who have played at least 100 matches

losers_pct= losers[['pct_home_defeats', 'pct_away_defeats', 'tot_pct_defeats']]





#total percentage

winners_pct.sort_values(by='tot_pct_victory', ascending=False)

winners_pct=np.round(winners_pct*100, 2)

winners_pct['tot_count']= winners.tot_count

winners_pct['tot_matches']= winners.tot_matches





losers_pct=np.round(losers_pct*100, 2)

losers_pct['tot_count']= losers.tot_count

losers_pct['tot_matches']= losers.tot_matches





winners_pct.sort_values(by='tot_pct_victory', ascending=False)



#discover goal and history



home_matches=match.groupby('home_team').sum().rename(columns={'home_score': 'goals_scored_home', 'away_score' : 'goals_taken_home'})



away_matches=match.groupby('away_team').sum().rename(columns={'home_score': 'goals_taken_away', 'away_score' : 'goals_scored_away'})



#merging 



score= pd.merge(home_matches, away_matches, how='inner', right_index= True, left_index=True)



#creating calculated fields

score['tot_goals_scored']= score['goals_scored_home'] + score['goals_scored_away']

score['tot_goals_taken']= score['goals_taken_home'] + score['goals_taken_away']

score['diff_goals_tot'] = score['tot_goals_scored'] - score['tot_goals_taken']

score.drop(columns=['neutral_x', 'goals_x', 'neutral_y', 'goals_y'], inplace=True)



#get the match count (only teams with 100 match played are taken)

score=pd.merge(score, winners[['home_matches', 'away_matches', 'tot_matches']], how='inner', right_index=True, left_index=True)



#compute ratios

score['home_goals_pct'] = score['goals_scored_home']/ score['home_matches']

score['away_goals_pct'] = score['goals_scored_away']/ score['away_matches']

score['home_goals_taken_pct'] = score['goals_taken_home']/ score['home_matches']

score['away_goals_taken_pct'] = score['goals_taken_away']/ score['away_matches']

score['tot_goals_scored_pct'] = score['tot_goals_scored']/ score['tot_matches']

score['tot_goals_taken_pct'] = score['tot_goals_taken']/ score['tot_matches']

score_pct= score[['home_goals_pct', 'away_goals_pct', 'home_goals_taken_pct', 'away_goals_taken_pct',

                 'tot_goals_scored_pct', 'tot_goals_taken_pct', 'diff_goals_tot']]



score_pct.sort_values(by='diff_goals_tot', ascending=False)



#join winners with score



total_ratio= pd.merge(score_pct, winners_pct, how='inner', right_index=True, left_index=True)

total_ratio= pd.merge(total_ratio, losers_pct, how='inner', right_index=True, left_index=True)



score_pct.sort_values(by='diff_goals_tot', ascending=False)



correlation= total_ratio[['tot_goals_scored_pct', 'tot_goals_taken_pct', 

                          'tot_pct_victory','tot_pct_defeats']].corr()





#machine learining to predict victory/loss rate

total_ratio.corr()['tot_pct_victory'] #check correlation

train = np.array(total_ratio.iloc[:, :7])

test= np.array(total_ratio['tot_pct_victory'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, test, random_state=42)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train, y_train)

lr.score(X_train,y_train)
lr.score(X_test, y_test)
from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor(n_estimators=200, max_features=2, max_depth=20, bootstrap=True, random_state=1)

rf.fit(X_train, y_train)

rf.score(X_train,y_train)
rf.score(X_test,y_test)
#goals through the years



year_goals= match.groupby(by=[match.index.year]).sum()['goals']

year_match= match.groupby(by=[match.index.year]).count()['neutral']

year=pd.DataFrame([year_goals, year_match]).T.rename(columns={'neutral': 'matches'})



#normalize

year['goals_match'] = year.goals/year.matches 

year=np.round(year, 2)




