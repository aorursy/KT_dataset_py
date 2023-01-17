import numpy as np 

import pandas as pd 

import IPython # to plot flourish 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Matches = pd.read_csv("/kaggle/input/international-football-results-from-1872-to-2017/results.csv")

Matches.date = pd.to_datetime(Matches.date)

Matches.info()
# Taking a look on dataset

Matches.sort_values('date')
#Applying the 3-1-0 points systems for win-draw-loss.

Matches.loc[Matches.home_score > Matches.away_score, "home_team_points"] = 3

Matches.loc[Matches.home_score < Matches.away_score, "away_team_points"] = 3

Matches.loc[Matches.home_score == Matches.away_score, "tie_points"] = 1

#Matches.fillna(0, inplace = True)
Home_win = Matches.loc[Matches.home_score > Matches.away_score][['date','home_team','home_team_points']]

Away_win = Matches.loc[Matches.home_score < Matches.away_score][['date','away_team','away_team_points']]

Home_tie = Matches.loc[Matches.home_score == Matches.away_score][['date','home_team','tie_points']]

Away_tie = Matches.loc[Matches.home_score == Matches.away_score][['date','away_team','tie_points']]



Home_win.columns = ['date', 'team', 'points']

Away_win.columns = ['date', 'team', 'points']

Home_tie.columns = ['date', 'team', 'points']

Away_tie.columns = ['date', 'team', 'points']
Points_over_time = pd.concat([Home_win,Away_win,Home_tie,Away_tie], ignore_index=True).sort_values('date', ascending = False)

Points_over_time["Year"] = Points_over_time.date.dt.year

Points_over_time
Alltimeleague = Points_over_time.groupby(['team','Year']).points.sum().unstack().fillna(0).T.cumsum().T

Alltimeleague
# Exporting csv to plot in flourish

Alltimeleague.to_csv("All_time_league.csv")

iframe = "<iframe src='https://flo.uri.sh/visualisation/2213639/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2213639/?utm_source=embed&utm_campaign=visualisation/2213639' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>"

IPython.display.HTML(iframe)