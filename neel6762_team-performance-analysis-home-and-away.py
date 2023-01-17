import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display, HTML
data = pd.read_csv('../input/results.csv')

data.head()
data.describe()
data.info()
# converting the season to a single year 

# 2006-2007 season is represented as 2006 season

arr = list(data['season'])

season = list()

for val in arr:

    temp = val.split('-')[0]

    season.append(temp)

season



season = pd.to_datetime(season)

season.year

data['season'] = season.year
data.head()
# Converting float to int

data['home_goals'] = data['home_goals'].astype(int)

data['away_goals'] = data['away_goals'].astype(int)

data.head()
data.info()
def get_team_home_stats(team,season):

    

    if(team in data['home_team'].values and season in data['season'].values):

        

        # Filetering the team

        mask1 = data['home_team'] == team

        mask2 = data['season'] == season

        team_stats = data[mask1 & mask2]

        

        print("------------------------------------ OVERALL HOME STATS --------------------------------------\n")

        print(team_stats.to_string(index=False))

        print("\n----------------------------------------------------------------------------------------------\n")

        

        # Home Wins

        mask = team_stats['result'] == 'H'

        home_wins = team_stats[mask]

        print("------------------------------------ HOME WINS ----------------------------------------\n")

        print(home_wins.to_string(index=False))

        print()

        goals_sc = home_wins['home_goals'].sum()

        print('Goals Scored : ',goals_sc)

        goals_cs = home_wins['away_goals'].sum()

        print('Goals Conceded : ',goals_cs)

        print("\n--------------------------------------------------------------------------------------\n")

        

        # Home Draws 

        mask = team_stats['result'] == 'D'

        home_draws = team_stats[mask]

        print("------------------------------------ HOME DRAWS ---------------------------------------\n")

        print(home_draws.to_string(index=False))

        print()

        goals_sc = home_draws['home_goals'].sum()

        print('Goals Scored : ',goals_sc)

        goals_cs = home_draws['away_goals'].sum()

        print('Goals Conceded : ',goals_cs)

        print("\n---------------------------------------------------------------------------------------\n")

        

        # Home Losses

        mask = team_stats['result'] == 'A'

        home_lost = team_stats[mask]

        print("------------------------------------ HOME LOSSES ----------------------------------------\n")

        print(home_lost.to_string(index=False))

        print()

        goals_sc = home_lost['home_goals'].sum()

        print('Goals Scored : ',goals_sc)

        goals_cs = home_lost['away_goals'].sum()

        print('Goals Conceded : ',goals_cs)

        print("\n-----------------------------------------------------------------------------------------\n")

        

        # Bar Representation

        x = ["Wins","Draws","Losses"]

        y = [home_wins.shape[0], home_draws.shape[0], home_lost.shape[0]]

        plt.bar(x,y,color=['green', 'gray', 'red'])

        title = team + " (" + str(season) + ") Home Stats"

        plt.title(title)

        plt.show()

        

    else:

        return 'Opps No Data Available'

    
get_team_home_stats('Manchester United',2017)
get_team_home_stats('Manchester City',2017)
get_team_home_stats('Liverpool',2017)
get_team_home_stats('Chelsea',2017)
get_team_home_stats('Tottenham Hotspur',2017)
get_team_home_stats('Manchester United',2013)
get_team_home_stats('Manchester United',2011)
def get_team_away_stats(team,season):

    

    if(team in data['away_team'].values and season in data['season'].values):

        

        # Filetering the team

        mask1 = data['away_team'] == team

        mask2 = data['season'] == season

        team_stats = data[mask1 & mask2]

        

        print("------------------------------------ OVERALL AWAY STATS ----------------------------------------\n")

        print(team_stats.to_string(index=False))

        print("\n------------------------------------------------------------------------------------------------\n")

        

        # Away Wins

        mask = team_stats['result'] == 'A'

        away_wins = team_stats[mask]

        print("------------------------------------ AWAY WINS ----------------------------------------\n")

        print(away_wins.to_string(index=False))

        print()

        goals_sc = away_wins['away_goals'].sum()

        print('Goals Scored : ',goals_sc)

        goals_cs = away_wins['home_goals'].sum()

        print('Goals Conceded : ',goals_cs)

        print("\n---------------------------------------------------------------------------------------\n")

        

        # Away Draws 

        mask = team_stats['result'] == 'D'

        away_draws = team_stats[mask]

        print("------------------------------------ AWAY DRAWS ----------------------------------------\n")

        print(away_draws.to_string(index=False))

        print()

        goals_sc = away_draws['away_goals'].sum()

        print('Goals Scored : ',goals_sc)

        goals_cs = away_draws['home_goals'].sum()

        print('Goals Conceded : ',goals_cs)

        print("\n-----------------------------------------------------------------------------------------\n")

        

        # Home Losses

        mask = team_stats['result'] == 'H'

        away_lost = team_stats[mask]

        print("------------------------------------ AWAY LOSSES ----------------------------------------\n")

        print(away_lost.to_string(index=False))

        print()

        goals_sc = away_lost['away_goals'].sum()

        print('Goals Scored : ',goals_sc)

        goals_cs = away_lost['home_goals'].sum()

        print('Goals Conceded : ',goals_cs)

        print("\n-----------------------------------------------------------------------------------------\n")

        

        # Bar Representation

        x = ["Wins","Draws","Losses"]

        y = [away_wins.shape[0], away_draws.shape[0], away_lost.shape[0]]

        plt.bar(x,y,color=['green', 'gray', 'red'])

        title = team + " (" + str(season) + ") Away Stats"

        plt.title(title)

        plt.show()

        

    else:

        return 'Opps No Data Available'

    
get_team_away_stats('Manchester United',2017)
get_team_home_stats('Arsenal',2017)
get_team_away_stats('Arsenal',2017)
get_team_away_stats('Manchester United',2019)
get_team_home_stats('Real Madrid',2006)