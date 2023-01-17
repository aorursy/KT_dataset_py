import numpy as np

import pandas as pd
DATA_DIR = '../input/epl-stats-20192020/'

POSITIONS = np.array(range(1, 21))

RANDOM_STATE = 0
game_data = pd.read_csv(DATA_DIR + 'epl2020.csv')
game_data['scored'] = game_data['scored'].apply(lambda x: str(x))

game_data.sort_values(by=['date', 'h_a'], inplace=True)

game_data.reset_index(inplace=True, drop=True)



# Get game data

results = game_data.groupby(by=['Referee.x', 'date']).agg({'teamId' : ','.join, 'scored' : ','.join}).reset_index()

results['teamA'] = results['teamId'].apply(lambda x: x.split(',')[0])

results['teamB'] = results['teamId'].apply(lambda x: x.split(',')[1])

results['scoredA'] = results['scored'].apply(lambda x: x.split(',')[0]).astype('uint16')

results['scoredB'] = results['scored'].apply(lambda x: x.split(',')[1]).astype('uint16')



# Add additional columns

results.sort_values(by='date', inplace=True)

results.reset_index(inplace=True, drop=True)

results['teamAPosition'] = 0

results['teamBPosition'] = 0

results['teamARecentScored'] = 0

results['teamBRecentScored'] = 0

results['teamARecentConceded'] = 0

results['teamBRecentConceded'] = 0

results['teamARecentPoints'] = 0

results['teamBRecentPoints'] = 0

results = results[['date', 'teamA', 'scoredA', 'teamB', 'scoredB', 'teamAPosition', 'teamBPosition', 'teamARecentScored', 'teamBRecentScored', 'teamARecentConceded', 'teamBRecentConceded', 'teamARecentPoints', 'teamBRecentPoints']]



# Display

#results
class LeagueTable:

    def __init__(self, teams):

        self.teams = list(teams)

        self.table = pd.DataFrame(np.array([teams, [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams), [0] * len(self.teams)]).T, columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Pts'])

        self.positions = np.array(range(1, len(self.teams) + 1))

        self.sort_table()

        

    def sort_table(self):

        self.table.sort_values(by=['Team'], ascending=True, inplace=True, ignore_index=True)

        self.table.sort_values(by=['Pts', 'GD', 'F'], ascending=False, inplace=True, ignore_index=True)

        self.table.set_index(self.positions, inplace=True)

    

    def show_table(self):

        return self.table.head(len(self.teams))

    

    def add_result(self, team_a, scored_a, team_b, scored_b):

        # Team A

        self.table.loc[self.table['Team'] == team_a, 'P'] += 1

        self.table.loc[self.table['Team'] == team_a, 'W'] += 1 if int(scored_a) > int(scored_b) else 0

        self.table.loc[self.table['Team'] == team_a, 'D'] += 1 if int(scored_a) == int(scored_b) else 0

        self.table.loc[self.table['Team'] == team_a, 'L'] += 1 if int(scored_a) < int(scored_b) else 0

        self.table.loc[self.table['Team'] == team_a, 'F'] += int(scored_a)

        self.table.loc[self.table['Team'] == team_a, 'A'] += int(scored_b)

        self.table.loc[self.table['Team'] == team_a, 'GD'] += int(scored_a) - int(scored_b)

        self.table.loc[self.table['Team'] == team_a, 'Pts'] += 3 if int(scored_a) > int(scored_b) else 1 if int(scored_a) == int(scored_b) else 0

        # Team B

        self.table.loc[self.table['Team'] == team_b, 'P'] += 1

        self.table.loc[self.table['Team'] == team_b, 'W'] += 1 if int(scored_b) > int(scored_a) else 0

        self.table.loc[self.table['Team'] == team_b, 'D'] += 1 if int(scored_b) == int(scored_a) else 0

        self.table.loc[self.table['Team'] == team_b, 'L'] += 1 if int(scored_b) < int(scored_a) else 0

        self.table.loc[self.table['Team'] == team_b, 'F'] += int(scored_b)

        self.table.loc[self.table['Team'] == team_b, 'A'] += int(scored_a)

        self.table.loc[self.table['Team'] == team_b, 'GD'] += int(scored_b) - int(scored_a)

        self.table.loc[self.table['Team'] == team_b, 'Pts'] += 3 if int(scored_b) > int(scored_a) else 1 if int(scored_b) == int(scored_a) else 0

        # Reorder table

        self.sort_table()
table = LeagueTable(results['teamA'].unique())

table.show_table()
for index, row in results.iterrows():

    # Update features

    previous_games = game_data[:index*2]

    results.iloc[index, 5] = table.show_table()[table.show_table()['Team'] == row['teamA']].index[0].astype('uint16')

    results.iloc[index, 6] = table.show_table()[table.show_table()['Team'] == row['teamB']].index[0].astype('uint16')

    results.iloc[index, 7] = previous_games[previous_games['teamId'] == row['teamA']][-5:]['scored'].astype('uint16').sum()

    results.iloc[index, 8] = previous_games[previous_games['teamId'] == row['teamB']][-5:]['scored'].astype('uint16').sum()

    results.iloc[index, 9] = previous_games[previous_games['teamId'] == row['teamA']][-5:]['missed'].astype('uint16').sum()

    results.iloc[index, 10] = previous_games[previous_games['teamId'] == row['teamB']][-5:]['missed'].astype('uint16').sum()

    results.iloc[index, 11] = 3*previous_games[previous_games['teamId'] == row['teamA']][-5:]['wins'].astype('uint16').sum() + previous_games[previous_games['teamId'] == row['teamA']][-5:]['draws'].astype('uint16').sum()

    results.iloc[index, 12] = 3*previous_games[previous_games['teamId'] == row['teamB']][-5:]['wins'].astype('uint16').sum() + previous_games[previous_games['teamId'] == row['teamB']][-5:]['draws'].astype('uint16').sum()

    # Add result to table

    table.add_result(row['teamA'], row['scoredA'], row['teamB'], row['scoredB'])
happened_games = [('Aston Villa', 'Sheffield United', 0, 0), ('Man City', 'Arsenal', 3, 0),

                  ('Aston Villa', 'Chelsea', 1, 2), ('Bournemouth', 'Crystal Palace', 0, 2), ('Brighton', 'Arsenal', 2, 1), ('Everton', 'Liverpool', 0, 0), ('Man City', 'Burnley', 5, 0), ('Newcastle United', 'Sheffield United', 3, 0), ('Norwich', 'Southampton', 0, 3), ('Tottenham', 'Man Utd', 1, 1), ('Watford', 'Leicester', 1, 1), ('West Ham', 'Wolves', 0, 2),

                  ('Burnley', 'Watford', 1, 0), ('Chelsea', 'Man City', 2, 1), ('Leicester', 'Brighton', 0, 0), ('Liverpool', 'Crystal Palace', 4, 0), ('Man Utd', 'Sheffield United', 3, 0), ('Newcastle United', 'Aston Villa', 1, 1), ('Norwich', 'Everton', 0, 1), ('Southampton', 'Arsenal', 0, 2), ('Tottenham', 'West Ham', 2, 0), ('Wolves', 'Bournemouth', 1, 0),

                  ('Aston Villa', 'Wolves', 0, 1), ('Watford', 'Southampton', 1, 3), ('Crystal Palace', 'Burnley', 0, 1), ('Brighton', 'Man Utd', 0, 3), ('Arsenal', 'Norwich', 4, 0), ('Bournemouth', 'Newcastle United', 1, 4), ('Everton', 'Leicester', 2, 1), ('West Ham', 'Chelsea', 3, 2), ('Man City', 'Liverpool', 4, 0), ('Sheffield United', 'Tottenham', 3, 1),

                  ('Burnley', 'Sheffield United', 1, 1), ('Chelsea', 'Watford', 3, 0), ('Leicester', 'Crystal Palace', 3, 0), ('Liverpool', 'Aston Villa', 2, 0), ('Man Utd', 'Bournemouth', 5, 2), ('Newcastle United', 'West Ham', 2, 2), ('Norwich', 'Brighton', 0, 1), ('Southampton', 'Man City', 1, 0), ('Tottenham', 'Everton', 1, 0), ('Wolves', 'Arsenal', 0, 2),

                  ('Arsenal', 'Leicester', 1, 1), ('Aston Villa', 'Man Utd', 0, 3), ('Bournemouth', 'Tottenham', 0, 0), ('Brighton', 'Liverpool', 1, 3), ('Crystal Palace', 'Chelsea', 2, 3), ('Everton', 'Southampton', 1, 1), ('Man City', 'Newcastle United', 5, 0), ('Sheffield United', 'Wolves', 1, 0), ('Watford', 'Norwich', 2, 1), ('West Ham', 'Burnley', 0, 1),

                  ('Aston Villa', 'Crystal Palace', 2, 0), ('Bournemouth', 'Leicester', 4, 1), ('Brighton', 'Man City', 0, 5), ('Liverpool', 'Burnley', 1, 1), ('Man Utd', 'Southampton', 2, 2), ('Norwich', 'West Ham', 0, 4), ('Sheffield United', 'Chelsea', 3, 0), ('Tottenham', 'Arsenal', 2, 1), ('Watford', 'Newcastle United', 2, 1), ('Wolves', 'Everton', 3, 0),

                  ('Arsenal', 'Liverpool', 2, 1), ('Burnley', 'Wolves', 1, 1), ('Chelsea', 'Norwich', 1, 0), ('Crystal Palace', 'Man Utd', 0, 2), ('Everton', 'Aston Villa', 1, 1), ('Leicester', 'Sheffield United', 2, 0), ('Man City', 'Bournemouth', 2, 1), ('Newcastle United', 'Tottenham', 1, 3), ('Southampton', 'Brighton', 1, 1), ('West Ham', 'Watford', 3, 1)]

for game in happened_games:

    # Get win/draw/loss

    home_win = 1 if round(game[2]) > round(game[3]) else 0

    home_draw = 1 if round(game[2]) == round(game[3]) else 0

    home_loss = 1 if round(game[2]) < round(game[3]) else 0

    

    table.add_result(game[0], game[2], game[1], game[3])

    new_row_a = pd.DataFrame([['', '', '', '', '', '', '', '', game[2], game[3], '', '', '', home_win, home_draw, home_loss, '', '', game[0], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)

    new_row_b = pd.DataFrame([['', '', '', '', '', '', '', '', game[3], game[2], '', '', '', home_loss, home_draw, home_win, '', '', game[1], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)

    game_data = pd.concat([game_data, new_row_a, new_row_b], ignore_index=True)
table.show_table()
games_left = [('Norwich', 'Burnley'), ('Bournemouth', 'Southampton'), ('Tottenham', 'Leicester'), ('Brighton', 'Newcastle United'), ('Sheffield United', 'Everton'), ('Wolves', 'Crystal Palace'), ('Watford', 'Man City'), ('Aston Villa', 'Arsenal'), ('Man Utd', 'West Ham'), ('Liverpool', 'Chelsea'),

              ('Arsenal', 'Watford'), ('Burnley', 'Brighton'), ('Chelsea', 'Wolves'), ('Crystal Palace', 'Tottenham'), ('Everton', 'Bournemouth'), ('Leicester', 'Man Utd'), ('Man City', 'Norwich'), ('Newcastle United', 'Liverpool'), ('Southampton', 'Sheffield United'), ('West Ham', 'Aston Villa')]





games_left = pd.DataFrame(games_left, columns=['teamA', 'teamB'])

games_left.sample(5, random_state=RANDOM_STATE)
from sklearn.multioutput import MultiOutputRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
X_train = results[['teamA', 'teamB', 'teamAPosition', 'teamBPosition', 'teamARecentScored', 'teamBRecentScored', 'teamARecentConceded', 'teamBRecentConceded', 'teamARecentPoints', 'teamBRecentPoints']]

X_train.fillna(0, inplace=True)

Y_train = results[['scoredA', 'scoredB']]
# Returns 1 if the team won the Premier League in the previous five seasons

def is_previous_winner(team):

    winners = ['Man City', 'Chelsea', 'Leicester']

    return 1 if team in winners else 0



# Returns 1 if the team finished in the top 4 of the Premier League in the previous five seasons

def is_previous_top_4(team):

    top_4 = ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Man Utd', 'Leicester', 'Arsenal']

    return 1 if team in top_4 else 0



# Returns 1 if the team played at least one season in the Championship (second division) in the previous five seasons

def is_championship(team):

    championships = ['Bournemouth', 'Watford', 'Norwich', 'Burnley', 'Newcastle United', 'Brighton', 'Wolves', 'Sheffield United', 'Aston Villa']

    return 1 if team in championships else 0



X_train['teamAPrevWinner'] = X_train['teamA'].apply(lambda x: is_previous_winner(x))

X_train['teamBPrevWinner'] = X_train['teamB'].apply(lambda x: is_previous_winner(x))

X_train['teamAPrevTop4'] = X_train['teamA'].apply(lambda x: is_previous_top_4(x))

X_train['teamBPrevTop4'] = X_train['teamB'].apply(lambda x: is_previous_top_4(x))

X_train['teamAPrevChampionship'] = X_train['teamA'].apply(lambda x: is_championship(x))

X_train['teamBPrevChampionship'] = X_train['teamB'].apply(lambda x: is_championship(x))



X_train.drop(['teamA', 'teamB'], inplace=True, axis=1)
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_train[:,0:2] *= 2.5

X_train[:,2:8] *= 1.5
model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=3, weights='distance')).fit(X_train, Y_train)
for index, row in games_left.iterrows():

    # Get recent form features

    a_position = table.show_table()[table.show_table()['Team'] == row['teamA']].index[0].astype('uint16')

    b_position = table.show_table()[table.show_table()['Team'] == row['teamB']].index[0].astype('uint16')

    a_scored = game_data[game_data['teamId'] == row['teamA']][-5:]['scored'].astype('uint16').sum()

    b_scored = game_data[game_data['teamId'] == row['teamB']][-5:]['scored'].astype('uint16').sum()

    a_conceded = game_data[game_data['teamId'] == row['teamA']][-5:]['missed'].astype('uint16').sum()

    b_conceded = game_data[game_data['teamId'] == row['teamB']][-5:]['missed'].astype('uint16').sum()

    a_points = 3*game_data[game_data['teamId'] == row['teamA']][-5:]['wins'].astype('uint16').sum() + game_data[game_data['teamId'] == row['teamA']][-5:]['draws'].astype('uint16').sum()

    b_points = 3*game_data[game_data['teamId'] == row['teamB']][-5:]['wins'].astype('uint16').sum() + game_data[game_data['teamId'] == row['teamB']][-5:]['draws'].astype('uint16').sum()

    

    

    # Get extra features

    a_prev = is_previous_winner(row['teamA'])

    b_prev = is_previous_winner(row['teamB'])

    a_top4 = is_previous_top_4(row['teamA'])

    b_top4 = is_previous_top_4(row['teamB'])

    a_cham = is_championship(row['teamA'])

    b_cham = is_championship(row['teamB'])

    

    # Make game prediction

    X_pred = np.array([a_position, b_position, a_scored, b_scored, a_conceded, b_conceded, a_points, b_points, a_prev, b_prev, a_top4, b_top4, a_cham, b_cham]).reshape(1, -1)

    X_pred = scaler.transform(X_pred)

    X_pred[:,0:2] *= 2.5

    X_pred[:,2:8] *= 1.5

    goals = model.predict(X_pred)

    

    # Add result to the table

    table.add_result(row['teamA'], round(goals[0][0]), row['teamB'], round(goals[0][1]))

    

    # Get win/draw/loss

    home_win = 1 if round(goals[0][0]) > round(goals[0][1]) else 0

    home_draw = 1 if round(goals[0][0]) == round(goals[0][1]) else 0

    home_loss = 1 if round(goals[0][0]) < round(goals[0][1]) else 0

    

    # Save the result (updating recent form)

    new_row_a = pd.DataFrame([['', '', '', '', '', '', '', '', goals[0][0], goals[0][1], '', '', '', home_win, home_draw, home_loss, '', '', row['teamA'], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)

    new_row_b = pd.DataFrame([['', '', '', '', '', '', '', '', goals[0][1], goals[0][0], '', '', '', home_loss, home_draw, home_win, '', '', row['teamB'], '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']], columns=game_data.columns)

    game_data = pd.concat([game_data, new_row_a, new_row_b], ignore_index=True)



# Show the final table

table.show_table()