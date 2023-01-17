import pandas as pd

import numpy as np



subset_results = pd.read_csv('../input/subsetresults/subset-1819.csv')



teams = subset_results['HomeTeam'].unique().tolist()

teams.sort()

subset_results['HomeTeamPoints'] = np.where(subset_results['FTR']=='H', 3, np.where(subset_results['FTR']=='D', 1, 0))

subset_results['AwayTeamPoints'] = np.where(subset_results['FTR']=='A', 3, np.where(subset_results['FTR']=='D', 1, 0))



all_results = []

#loop through all teams and get the points they received per gameweek.

for team in teams:

    team_results_home = subset_results[subset_results['HomeTeam'].str.match(team)] #get results where team is at home

    team_results_away = subset_results[subset_results['AwayTeam'].str.match(team)] #get results where team is away

    team_results = pd.concat([team_results_home, team_results_away]) #bring together into one list

    team_results = team_results.sort_values('GameWeek') #order list by gameweek to create points per game list

    team_results['Points'] = np.where(team_results['HomeTeam']== team,team_results['HomeTeamPoints'], team_results['AwayTeamPoints']) #turn home and away points into one column of Points

    team_results['Cumulative Points'] = team_results['Points'].cumsum(axis=0) #create additional column for cumulative points

    team_points = team_results['Cumulative Points'].tolist() #take cumulative points column to new variable as a list

    all_results.append(team_points) #append team points list to master list



np.array(all_results)

columns = list(range(1, 39))



results = pd.DataFrame(columns = columns, data = all_results)

results.insert(0, 'Team', teams)

results_teams_as_index = results.set_index(['Team'])

print(results_teams_as_index)
from matplotlib import pyplot as plt



plt.figure(figsize =(15,10))

ax = plt.subplot()

for i in range(len(results_teams_as_index)):

    plt.plot(columns, results_teams_as_index.iloc[i], marker='o')



plt.title('English Premier League 2018/19 season - Cumulative points by games played')

plt.xlabel('Games played')

plt.ylabel('Total Points')

plt.legend(teams, loc='best')

ax.set_xticks(columns)

ax.set_xticklabels(columns)

ax.set_yticks(range(0, 100, 5))

plt.show()
function_results = pd.read_csv('../input/results/results.csv')



def create_table(gameweek):

    table = function_results.sort_values(by=[gameweek], ascending=False)

    table = table.reset_index(drop=True)

    table.index = table.index + 1

    current_table = table[['Team', gameweek]]

    current_table.columns = ['Team', 'Points']

    return print(current_table)



create_table('21')
team_results_csv = pd.read_csv('../input/subsetresults/subset-1819.csv')



teams = team_results_csv['HomeTeam'].unique().tolist()

teams.sort()

team_results_csv['HomeTeamPoints'] = np.where(team_results_csv['FTR']=='H', 3, np.where(team_results_csv['FTR']=='D', 1, 0))

team_results_csv['AwayTeamPoints'] = np.where(team_results_csv['FTR']=='A', 3, np.where(team_results_csv['FTR']=='D', 1, 0))



all_results = []

#loop through all teams and get the points they received per gameweek.

for team in teams:

    team_results_home = team_results_csv[subset_results['HomeTeam'].str.match(team)] #get results where team is at home

    team_results_away = team_results_csv[subset_results['AwayTeam'].str.match(team)] #get results where team is away

    team_results = pd.concat([team_results_home, team_results_away]) #bring together into one list

    team_results = team_results.sort_values('GameWeek') #order list by gameweek to create points per game list

    team_results['Points'] = np.where(team_results['HomeTeam']== team,team_results['HomeTeamPoints'], team_results['AwayTeamPoints']) #turn home and away points into one column of Points

    team_points = team_results['Points'].tolist() 

    all_results.append(team_points) #append team points list to master list



np.array(all_results)

columns = list(range(1, 39))



points = pd.DataFrame(columns = columns, data = all_results)

points.insert(0, 'Team', teams)

points_teams_as_index = points.set_index(['Team'])

print(points_teams_as_index)


def create_table_from(gwstart, gwend):

    subset_table = points.iloc[:, gwstart:gwend]

    subset_table['Points'] = subset_table.sum(axis=1)

    subset_table.insert(0, 'Team', teams)

    subset_table = subset_table.sort_values(by=['Points'], ascending=False)

    subset_table = subset_table.reset_index(drop=True)

    subset_table.index = subset_table.index + 1

    current_table = subset_table[['Team', 'Points']]

    return print(current_table)



create_table_from(20, 30)