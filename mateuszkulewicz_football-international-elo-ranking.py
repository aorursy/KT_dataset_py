# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



data = pd.read_csv('/kaggle/input/international-football-results-from-1872-to-2017/results.csv')

ranking = pd.DataFrame(columns=['date', 'team', 'score_change', 'current_score'])



for match in data.itertuples():

    print(match)

    if not match.home_team in ranking["team"].to_numpy():

        ranking = ranking.append(pd.DataFrame({

            'date': [match.date],

            'team': [match.home_team],

            'score_change': [1000.0],

            'current_score': [1000.0]

        }))

    if not match.away_team in ranking["team"].to_numpy():

        ranking = ranking.append(pd.DataFrame({

            'date': [match.date],

            'team': [match.away_team],

            'score_change': [1000.0],

            'current_score': [1000.0]

        }))

    home_team_rank = ranking.loc[ranking['team'] == match.home_team, 'score_change'].sum()

    away_team_rank = ranking.loc[ranking['team'] == match.away_team, 'score_change'].sum()

    home_score = match.home_score

    away_score = match.away_score

    goals = float(home_score + away_score)

    home_contributed_points = home_team_rank * (goals + 1) / 100

    away_contributed_points = away_team_rank * (goals + 1) / 100

    points = home_contributed_points + away_contributed_points

    if (goals < 0.5):

        home_won_points = points / 2

        home_diff = home_won_points - home_contributed_points

        home_new_score = home_team_rank + home_diff

        away_won_points = points / 2

        away_diff = away_won_points - away_contributed_points

        away_new_score = away_team_rank + away_diff

        ranking = ranking.append(pd.DataFrame({

            'date': [match.date],

            'team': [match.home_team],

            'score_change': [home_diff],

            'current_score': [home_new_score]

        }))

        ranking = ranking.append(pd.DataFrame({

            'date': [match.date],

            'team': [match.away_team],

            'score_change': [away_diff],

            'current_score': [away_new_score]

        }))

    else:

        home_won_points = points * home_score / goals

        home_diff = home_won_points - home_contributed_points

        home_new_score = home_team_rank + home_diff

        away_won_points = points * away_score / goals

        away_diff = away_won_points - away_contributed_points

        away_new_score = away_team_rank + away_diff

        ranking = ranking.append(pd.DataFrame({

            'date': [match.date],

            'team': [match.home_team],

            'score_change': [home_diff],

            'current_score': [home_new_score]

        }))

        ranking = ranking.append(pd.DataFrame({

            'date': [match.date],

            'team': [match.away_team],

            'score_change': [away_diff],

            'current_score': [away_new_score]

        }))

    

print(ranking)

ranking.to_csv('full_ranking.csv',index=False)

        