# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#create a dataset called 'games' with the 'spreadspoke_scores.csv file'

games = pd.read_csv('/kaggle/input/nfl-scores-and-betting-data/spreadspoke_scores.csv')
#print out the final 20 rows of dataset 'games' using the .tail() function.

games.tail(20)
#new dataset named 'games_2010s' to hold all games from 2010-2019. The data is populated by adding the correct games from the 'games' dataset

games_2010s = games[ games.schedule_season > 2009]



#print out the first five games of the years 2010-2019

games_2010s.head()
#print out the number of rows in the dataset 'games_2010'

games_2010s.shape[0]
#create groups based on the 'weather_detail' column and counts the number of rows with that value in the column.'

games_2010s.groupby('weather_detail').size()
#creates a new column called total points and appends the total points scored by the home and away teams of each game.

games_2010s['total_points'] = games_2010s.score_home + games_2010s.score_away
#create the function and apply it to dataframes.



def indoor_outdoor_test(df):

    

#use if statement to conditionally select rows that have 'DOME' or 'DOME (Open Roof) to return 'Indoors'

    

    if df['weather_detail'] == 'DOME':

        return 'Indoors'

    elif df['weather_detail'] == 'DOME (Open Roof)':

        return 'Indoors'

    

#every other row should return 'Outdoors'

    

    else:

        return 'Outdoors'



#create a new column for the dataset 'games_2010s' and apply the indoor_outdoor_test function to the dataset.

games_2010s['indoor_outdoor'] = games_2010s.apply(indoor_outdoor_test, axis=1)
#create vertical bar chart of average total points per game based on indoor and outdoor games

games_2010s.groupby('indoor_outdoor').total_points.mean().plot.bar()
#create horizontal bar chart of average total points per game based on indoor and outdoor games

games_2010s.groupby('indoor_outdoor').total_points.median().plot.barh()
#create a dataset that just contains the games from the 2019 season

games_2019 = games_2010s[ games_2010s.schedule_season == 2019]



#create a dataset that removes the Super Bowl from the dataset due to its being played on a neutral site game

regular_season = games_2019[ games_2019.stadium_neutral == False]



#create separate datasets that hold the Saints' home games and away games

NOS_2019_home = regular_season[ regular_season.team_home == 'New Orleans Saints']

NOS_2019_away = regular_season[ regular_season.team_away == 'New Orleans Saints']



#print out the Saints' home games in order of highest to lowest point total

NOS_2019_home.sort_values('total_points', ascending=False)
NOS_2019_away.sort_values('total_points', ascending=True)
NOS_2019_home.total_points.mean()
NOS_2019_away.total_points.mean()