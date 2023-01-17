import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
DATA_DIR = '../input/epl-stats-20192020/'

POSITIONS = np.array(range(1, 21))
game_data = pd.read_csv(DATA_DIR + 'epl2020.csv')

print(game_data.columns)

game_data.head(10)
# Get the list of teams

teams = game_data['teamId'].unique()



# Get the results for each team

team_results = []

for team in teams:

    # Get the data for that team

    team_data = game_data[game_data['teamId'] == team]

    

    wins = team_data['wins'].sum()

    draws = team_data['draws'].sum()

    losses = team_data['loses'].sum()

    scored = team_data['scored'].sum()

    conceded = team_data['missed'].sum()

    games = wins + draws + losses

    points = (3 * wins) + draws

    goal_difference = scored - conceded

    

    team_results.append([team, games, wins, draws, losses, scored, conceded, goal_difference, points])



league_table = pd.DataFrame(team_results, columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Points'])

league_table.sort_values(by=['Points', 'GD', 'F'], ascending=False, inplace=True, ignore_index=True)

league_table.set_index(POSITIONS, inplace=True)

league_table.head(20)
# Get a rounded expected goals scored and conceded

game_data['xGround'] = game_data['xG'].apply(lambda x: round(x))

game_data['xGAround'] = game_data['xGA'].apply(lambda x: round(x))

game_data['xwin'] = game_data.apply(lambda x: 1 if x['xGround'] > x['xGAround'] else 0, axis=1)

game_data['xdraw'] = game_data.apply(lambda x: 1 if x['xGround'] == x['xGAround'] else 0, axis=1)

game_data['xloss'] = game_data.apply(lambda x: 1 if x['xGround'] < x['xGAround'] else 0, axis=1)
# Get the results for each team

x_team_results = []

for team in teams:

    # Get the data for that team

    team_data = game_data[game_data['teamId'] == team]

    

    wins = team_data['xwin'].sum()

    draws = team_data['xdraw'].sum()

    losses = team_data['xloss'].sum()

    scored = team_data['xGround'].sum()

    conceded = team_data['xGAround'].sum()

    games = wins + draws + losses

    points = (3 * wins) + draws

    goal_difference = scored - conceded

    

    x_team_results.append([team, games, wins, draws, losses, scored, conceded, goal_difference, points])



x_league_table = pd.DataFrame(x_team_results, columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Points'])

x_league_table.sort_values(by=['Points', 'GD', 'F'], ascending=False, inplace=True, ignore_index=True)

x_league_table.set_index(POSITIONS, inplace=True)

x_league_table.head(20)
# Get the results for each team

xp_team_results = []

for team in teams:

    # Get the data for that team

    team_data = game_data[game_data['teamId'] == team]

    

    xp = team_data['xpts'].sum()

    

    xp_team_results.append([team, xp])



xp_league_table = pd.DataFrame(xp_team_results, columns=['Team', 'Points'])

xp_league_table.sort_values(by=['Points'], ascending=False, inplace=True, ignore_index=True)

xp_league_table.set_index(POSITIONS, inplace=True)

xp_league_table.head(20)
team_positions = []

for team in teams:

    current_pos = league_table[league_table['Team'] == team].index[0]

    xg_pos = x_league_table[x_league_table['Team'] == team].index[0]

    xp_pos = xp_league_table[xp_league_table['Team'] == team].index[0]

    overperforming = 'Yes' if current_pos < xg_pos else 'No'

    team_positions.append([team, current_pos, xg_pos, xp_pos, overperforming])



position_table = pd.DataFrame(team_positions, columns=['Team', 'Position', 'xG Position', 'xPts Position', 'Overperforming'])

position_table.sort_values(by=['Position'], ascending=True, inplace=True, ignore_index=True)

position_table.head(20)
recent_form = []



for team in teams:

    # Get the data for that team

    team_data = game_data[game_data['teamId'] == team].tail(6)

    

    wins = team_data['wins'].sum()

    draws = team_data['draws'].sum()

    points = (3 * wins) + draws

    

    recent_form.append([team, points])



recent_form.sort(key=lambda x: x[1])



plt.figure(figsize = (8, 8))

plt.barh(range(20), [x[1] for x in recent_form])

plt.xlabel('Points')

plt.ylabel('Team')

plt.title('Points In Last 6 Games')

plt.yticks(range(20), [x[0] for x in recent_form])

plt.show()
team_points = []



# Add recent points per game to table

for team in teams:

    points_per_game = [x for x in recent_form if x[0] == team][0][1] / 6

    team_data = league_table[league_table['Team'] == team].iloc[0]

    games_to_play = 38 - team_data['P']

    new_points = int(team_data['Points'] + round(points_per_game * games_to_play))

    team_points.append([team, new_points])



predicted_table = pd.DataFrame(team_points, columns=['Team', 'Points'])

predicted_table.sort_values(by=['Points'], ascending=False, inplace=True, ignore_index=True)

predicted_table.set_index(POSITIONS, inplace=True)

predicted_table.head(20)