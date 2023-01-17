import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import regex as re
import os
from pathlib import Path

spreadspoke_path = Path("../input", 'spreadspoke_scores.csv')
nfl_games = pd.read_csv(spreadspoke_path)

# Drop columns with
    # sparse data entries: weather_detail, weather_humidity
    # sparse data entries for the superbowl: weather_temperature, weather_wind_mph
    # heavily unbalanced categorical data: stadium_neutral, schedule_playoff
    # Unused columns: stadium
nfl_games.drop(['weather_detail', 'weather_humidity'],axis='columns',inplace=True)
nfl_games.drop(['weather_temperature', 'weather_wind_mph'],axis='columns',inplace=True)
nfl_games.drop(['stadium_neutral', 'schedule_playoff'],axis='columns',inplace=True)
nfl_games.drop(['schedule_date','stadium','over_under_line'],axis='columns',inplace=True)

# Drop rows for games before year 1980
nfl_games_since_1980_bool = (nfl_games.loc[:,'schedule_season'] >= 1980)
nfl_games_since_1980 = nfl_games[nfl_games_since_1980_bool]

# Drop rows for games in the future
game_was_played_bool = nfl_games_since_1980.loc[:,'score_home'].notnull()
nfl_games_since_1980 = nfl_games_since_1980[game_was_played_bool].copy()
nfl_games_since_1980.head()
nfl_games_since_1980.team_favorite_id.unique()
# Need to replace the above team IDs in team_favorite_id 
# so we can match the team_home/team_away with team_favorite_id
name_mappings = [('ATL','Falcons'),
                 ('BUF', 'Bills'),
                 ('JAX','Jaguars'),
                 ('DAL', 'Cowboys'),
                 ('GB', 'Packers'),
                 ('IND','Colts'),
                 ('MIA', 'Dolphins'),
                 ('MIN','Vikings'),
                 ('TB', 'Buccaneers'),
                 ('NYG','Giants'),
                 ('OAK', 'Raiders'),
                 ('BAL','Ravens'),
                 ('WAS', 'Redskins'),
                 ('LAR', 'Rams'),
                 ('ARI', 'Cardinals'),
                 ('CIN', 'Bengals'),
                 ('DEN', 'Broncos'),
                 ('PHI', 'Eagles'),
                 ('LAC','Chargers'),
                 ('CAR','Panthers'),
                 ('TEN','Titans'),
                 ('NYJ', 'Jets'),
                 ('CHI', 'Bears'),
                 ('PIT', 'Steelers'),
                 ('KC', 'Chiefs'),
                 ('NE','Patriots'),
                 ('SEA','Seahawks'),
                 ('NO', 'Saints'),
                 ('DET', 'Lions'),
                 ('SF', '49ers'),
                 ('CLE', 'Browns'),
                 ('HOU', 'Texans')]
    
for mapping in name_mappings:
    nfl_games_since_1980.team_favorite_id.replace(mapping[0], mapping[1],inplace=True)
nfl_games_since_1980.head()
# Function determines if home or away team is favorite and returns how much the favorite won by
def calc_favorite_won_by (row):
    # Favored team played at home
    if re.search(row['team_favorite_id'], row['team_home'], re.IGNORECASE):
        return row['score_home']-row['score_away']
    else:
        return row['score_away']-row['score_home']

# Function determines if home or away team is favorite and returns how much the favorite beat spread by
def calc_favorite_beat_spread_by (row):
    return row['spread_favorite']+row['favorite_won_by']    
    
# Populate new columns with how much the favorite won by and how much the favorite beat spread by
nfl_games_since_1980['favorite_won_by']=nfl_games_since_1980.apply(lambda row: calc_favorite_won_by(row),axis=1)
nfl_games_since_1980['favorite_beat_spread_by']=nfl_games_since_1980.apply(lambda row: calc_favorite_beat_spread_by(row),axis=1)

nfl_games_since_1980.head()
nfl_games_since_1980.isnull().sum()
print("Favorite beat spread")
print((nfl_games_since_1980.loc[:,'favorite_won_by']+nfl_games_since_1980.loc[:,'spread_favorite']>0).value_counts(True))
print("\nFavorite won")
print((nfl_games_since_1980.loc[:,'favorite_won_by']>0).value_counts(True))
is_superbowl_since_1980 = (nfl_games_since_1980.loc[:,'schedule_week'].str.lower()=="superbowl")
print("\nFavorite beat spread in Superbowl")
print((nfl_games_since_1980.loc[is_superbowl_since_1980,'favorite_won_by']+nfl_games_since_1980.loc[is_superbowl_since_1980,'spread_favorite']>0).value_counts(True))
print("\nFavorite won in Superbowl")
print((nfl_games_since_1980.loc[is_superbowl_since_1980,'favorite_won_by']>0).value_counts(True))
superbowls_since_1980 = nfl_games_since_1980.loc[is_superbowl_since_1980].copy()

superbowls_since_1980.head()
superbowls_since_1980.head()
# Takes in nfl games df and a year and
# downselects to games in the specified year
def downselect_to_games_in_year(nfl_df,year):
    in_year_only_bool = (nfl_df.loc[:,'schedule_season'] == year)
    return nfl_df.loc[in_year_only_bool].copy()
# Takes in games from a single year and 
# returns lists of Super Bowl teams (city + team name) and Super Bowl team names
def get_superbowl_teams(nfl_year_only):
    # Find the teams in the Superbowl that year
    is_superbowl_for_year = (nfl_year_only.loc[:,'schedule_week'].str.lower()=="superbowl")
    superbowl_teams_df = nfl_year_only.loc[is_superbowl_for_year,['team_home','team_away']].copy()
    superbowl_teams = list(superbowl_teams_df.values.flatten())

    # Extract team names from longer strings
    superbowl_team_names = []
    for i,team in enumerate(superbowl_teams):
        team_name = team.split(' ')[-1]
        if team_name not in superbowl_team_names:
            superbowl_team_names.append(team.split(' ')[-1])
    
    return (superbowl_teams,superbowl_team_names)
# Takes in games from a single year and 
# returns the list of playoff teams
def get_playoff_teams(nfl_year_only):
    # Find the teams in the Super Bowl that year
    is_playoff_in_year = (nfl_year_only.loc[:,'schedule_week'].str.lower().isin(["superbowl","division","conference"]))
    playoff_teams_df = nfl_year_only.loc[is_playoff_in_year,['team_home','team_away']].copy()
    playoff_teams = list(playoff_teams_df.values.flatten())
    
    return playoff_teams
# Takes in games from a single year and list of Super Bowl teams and
# downselects to games that include either Super Bowl team
def downselect_to_games_with_superbowl_teams(nfl_year_only, superbowl_teams):
    team_home_in_superbowl_bool = (nfl_year_only['team_home'].isin(superbowl_teams)) 
    team_away_in_superbowl_bool = (nfl_year_only['team_away'].isin(superbowl_teams))
    team_in_superbowl_bool = team_home_in_superbowl_bool | team_away_in_superbowl_bool

    return nfl_year_only.loc[team_in_superbowl_bool].copy()
# Takes in games with Super Bowl teams and
# returns superbowl favorite's ID for this year
def get_superbowl_favorite_id(games_with_superbowl_teams):
    is_superbowl_for_year = (games_with_superbowl_teams.loc[:,'schedule_week'].str.lower()=="superbowl")
    superbowl_favorite_id = games_with_superbowl_teams.loc[is_superbowl_for_year,'team_favorite_id'].copy()
    superbowl_favorite_id = superbowl_favorite_id.values[0]
    return superbowl_favorite_id
# Populates has_superbowl_favorite column
def populate_has_superbowl_favorite_column(games_with_superbowl_teams,superbowl_favorite_id):
    has_superbowl_favorite_bool = games_with_superbowl_teams['team_home'].str.contains(superbowl_favorite_id,0) | games_with_superbowl_teams['team_away'].str.contains(superbowl_favorite_id,0)
    games_with_superbowl_teams.loc[:,'has_superbowl_favorite'] = has_superbowl_favorite_bool.astype(int).copy()
    return games_with_superbowl_teams
# Populates favored_team_in_superbowl column
def populate_favored_team_in_superbowl(games_with_superbowl_teams,superbowl_team_names):
    favored_team_in_superbowl_bool = games_with_superbowl_teams['team_favorite_id'].str.contains(superbowl_team_names[0],0) | games_with_superbowl_teams['team_favorite_id'].str.contains(superbowl_team_names[1],0)
    games_with_superbowl_teams.loc[:,'favored_team_in_superbowl'] = favored_team_in_superbowl_bool.astype(int).copy()
    return games_with_superbowl_teams
# Takes in games from a single year and the list of playoff teams and
# downselects to games against playoff teams
def downselect_to_games_against_playoff_teams(nfl_year_only, playoff_teams):
    # Downselect to games against playoff teams
    team_home_in_playoffs_bool = (nfl_year_only['team_home'].isin(playoff_teams))  
    team_away_in_playoffs_bool = (nfl_year_only['team_away'].isin(playoff_teams))
    team_in_playoffs_bool = team_home_in_playoffs_bool & team_away_in_playoffs_bool

    return nfl_year_only.loc[team_in_playoffs_bool].copy()
# Takes in the year
# returns games with Super Bowl teams, games with playoff teams, 
# Super Bowl favorite ID, and list of Super Bowl teams
def extract_info_from_nfl_df_for_year(year):
    # Downselect to this year's games
    nfl_year_only = downselect_to_games_in_year(nfl_games_since_1980, year)

    # For this year's games
    # Get list of Super Bowl teams, list of Super Bowl team names,
    # list of playoff teams, & games with Super Bowl teams df
    (superbowl_teams_list,superbowl_team_names_list) = get_superbowl_teams(nfl_year_only)
    
    playoff_teams_list = get_playoff_teams(nfl_year_only)
    
    games_with_superbowl_teams = downselect_to_games_with_superbowl_teams(nfl_year_only, superbowl_teams_list)

    # For this year's games with Super Bowl teams
    # Get Super Bowl favorite ID, populate additional columns in
    # games with Super Bowl teams df, and games with playoff teams df
    superbowl_favorite_id = get_superbowl_favorite_id(games_with_superbowl_teams)
    
    games_with_superbowl_teams = populate_has_superbowl_favorite_column(games_with_superbowl_teams, superbowl_favorite_id)

    games_with_superbowl_teams = populate_favored_team_in_superbowl(games_with_superbowl_teams, superbowl_team_names_list)
    
    games_with_playoff_teams = downselect_to_games_against_playoff_teams(games_with_superbowl_teams, playoff_teams_list)
    
    return (games_with_superbowl_teams, games_with_playoff_teams, superbowl_favorite_id, superbowl_teams_list)

for year in range(1980,2018):
    (games_with_superbowl_teams, games_with_playoff_teams, superbowl_favorite_id, superbowl_teams) = extract_info_from_nfl_df_for_year(year)
    
    # Determine super bowl favorite's team name
    if superbowl_favorite_id in superbowl_teams[0]:
        superbowl_favorite = superbowl_teams[0]
        superbowl_unfavorite = superbowl_teams[1]
    else:
        superbowl_favorite = superbowl_teams[1]
        superbowl_unfavorite = superbowl_teams[0]

    # Remove superbowl from df and separate games into (2) dfs for
    # games with Super Bowl favorite & games with Super Bowl unfavorite
    not_superbowl_bool = (games_with_superbowl_teams.loc[:,'schedule_week'].str.lower()!="superbowl")
    game_has_superbowl_favorite_bool = (games_with_superbowl_teams.loc[:,'has_superbowl_favorite']==1) & not_superbowl_bool
    # e.g. Pats games
    all_games_with_superbowl_favorite = games_with_superbowl_teams.loc[game_has_superbowl_favorite_bool].copy()
    # e.g. Eagles games
    all_games_with_superbowl_unfavorite = games_with_superbowl_teams.loc[~game_has_superbowl_favorite_bool & not_superbowl_bool].copy()
    
    # Calculate points_against_favorite
    favored_team_in_superbowl_is_hometeam = all_games_with_superbowl_favorite.loc[:,'team_home'] == superbowl_favorite
    points_against_favorite = (all_games_with_superbowl_favorite.loc[favored_team_in_superbowl_is_hometeam,'score_away'].sum() + 
                               all_games_with_superbowl_favorite.loc[~favored_team_in_superbowl_is_hometeam,'score_home'].sum())
    
    # Calculate wins_for_favorite
    home_team_wins = all_games_with_superbowl_favorite.loc[:,'score_home'] > all_games_with_superbowl_favorite.loc[:,'score_away']
    wins_for_favorite = (all_games_with_superbowl_favorite.loc[favored_team_in_superbowl_is_hometeam & home_team_wins,'score_home'].size + 
                          all_games_with_superbowl_favorite.loc[~favored_team_in_superbowl_is_hometeam & ~home_team_wins,'score_away'].size)
    
    # Calculate points_for_unfavorite
    unfavored_team_in_superbowl_is_hometeam = all_games_with_superbowl_unfavorite.loc[:,'team_home'] == superbowl_unfavorite
    points_for_unfavorite = (all_games_with_superbowl_unfavorite.loc[unfavored_team_in_superbowl_is_hometeam,'score_home'].sum() + 
                                 all_games_with_superbowl_unfavorite.loc[~unfavored_team_in_superbowl_is_hometeam,'score_away'].sum())    

    # Add features to superbowls_since_1980 df
    is_superbowl_in_year = (superbowls_since_1980.loc[:,'schedule_season'] == year)
    superbowls_since_1980.loc[is_superbowl_in_year,'points_against_favorite'] = points_against_favorite.copy()
    superbowls_since_1980.loc[is_superbowl_in_year,'wins_for_favorite'] = wins_for_favorite
    superbowls_since_1980.loc[is_superbowl_in_year,'points_for_unfavorite'] = points_for_unfavorite.copy()
    superbowls_since_1980.loc[:,'unfavorite_coach_win_%'] = [0.75,0.75,0.75,1,0.58,0.75,0.67,0.77,0.75,0.6, 0.7,0.625,0.67,0.67,0.75,0.5,0.71,0.75,0.53,0.75,0.67,0.75,0.67,0.75,0.78,0.58,0.67,0.54,0.75,0.75,0.86,0.61,0.67,0.56,0.67,0.67,0.75,0.75]
superbowls_since_1980.tail()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics

feature_cols = [ 'points_against_favorite',   # favorite's lack of defense loses to spread
                 'wins_for_favorite',         # favorite's good record beats spread
                 'unfavorite_coach_win_%']    # underdogs winning coach beats spread

#                 'points_for_unfavorite',     # underdogs offense beats spread // strange positive correlation

X = superbowls_since_1980.loc[:,feature_cols]
y = superbowls_since_1980.loc[:,'favorite_beat_spread_by']

loo = LeaveOneOut()
loo.get_n_splits(X)

MSE_train_scores = []
MSE_scores = []
null_MSE_scores = []

for train_indices, test_indices in loo.split(X, y):
    lr = LinearRegression()
    X_train = X.iloc[train_indices, :]
    y_train = y.iloc[train_indices]
    lr.fit(X_train, y_train)
    
    # Calculate training MSE
    y_train_pred = lr.predict(X_train)
    MSE_train = metrics.mean_squared_error(y_train,y_train_pred)
    MSE_train_scores.append(MSE_train)
    
    # Calculate MSE
    X_test = X.iloc[test_indices, :]
    y_test = y.iloc[test_indices]
    y_pred = lr.predict(X_test)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    MSE_scores.append(MSE)
    
    # Calculate null MSE
    y_pred_null = np.zeros(y_test.shape) + y_train.mean()
    null_MSE = metrics.mean_squared_error(y_test,y_pred_null)
    null_MSE_scores.append(null_MSE)

print('max MSE:',max(MSE_scores))
print('MSE:', np.array(MSE_scores).mean())
print('null MSE:', np.array(null_MSE_scores).mean()) 
print('MSE train:', np.array(MSE_train))
# Use 1980 to 2016 Super Bowls to predict 2017 Super Bowl
lr_point_spread = LinearRegression()
X_train = X.iloc[:-1,:]
y_train = y.iloc[:-1]
X_test = X.tail(1)
y_test = y.tail(1)

lr_point_spread.fit(X_train,y_train)

y_pred = lr_point_spread.predict(X_test)
MSE = metrics.mean_squared_error(y_test,y_pred)

print('intercept:', lr_point_spread.intercept_)
print('coefficients:', lr_point_spread.coef_)
print('y_pred:',y_pred[0])
print('y_test:',y_test.values[0])
print('MSE:',MSE)
import warnings
warnings.filterwarnings("ignore")

superbowls_since_1980.loc[:,'points_against_favorite'] = pd.to_numeric(superbowls_since_1980.loc[:,'points_against_favorite'])
x = superbowls_since_1980.loc[:,'points_against_favorite']
y = superbowls_since_1980.loc[:,'favorite_beat_spread_by']
g = sns.lmplot(x='points_against_favorite',y='favorite_beat_spread_by',data=superbowls_since_1980)
print(np.corrcoef(x, y))

x = superbowls_since_1980.loc[:,'wins_for_favorite']
y = superbowls_since_1980.loc[:,'favorite_beat_spread_by']
g = sns.lmplot(x='wins_for_favorite',y='favorite_beat_spread_by',data=superbowls_since_1980)
print(np.corrcoef(x, y))

x = superbowls_since_1980.loc[:,'unfavorite_coach_win_%']
y = superbowls_since_1980.loc[:,'favorite_beat_spread_by']
g = sns.lmplot(x='unfavorite_coach_win_%',y='favorite_beat_spread_by',data=superbowls_since_1980)
print(np.corrcoef(x, y))

x = superbowls_since_1980.loc[:,'points_for_unfavorite']
y = superbowls_since_1980.loc[:,'favorite_beat_spread_by']
g = sns.lmplot(x='points_for_unfavorite',y='favorite_beat_spread_by',data=superbowls_since_1980)
print(np.corrcoef(x, y))
feature_cols = ['points_against_favorite','unfavorite_coach_win_%', 'wins_for_favorite']
ax2 = sns.heatmap(superbowls_since_1980.loc[:,feature_cols].corr(),vmin=-1,vmax=1,cmap=sns.diverging_palette(h_neg=220,h_pos=10,n=21))
import seaborn as sns; sns.set(style="ticks", color_codes=True)
ax1 = sns.pairplot(superbowls_since_1980.loc[:,feature_cols])

#     not_superbowl_bool = (games_with_playoff_teams.loc[:,'schedule_week'].str.lower()!="superbowl")
#     game_has_superbowl_favorite_bool = (games_with_playoff_teams.loc[:,'has_superbowl_favorite']==1) & not_superbowl_bool
#     # e.g. Pats games
#     games_with_superbowl_favorite = games_with_playoff_teams.loc[game_has_superbowl_favorite_bool].copy()
#     # e.g. Eagles games
#     games_with_superbowl_unfavorite = games_with_playoff_teams.loc[~game_has_superbowl_favorite_bool & not_superbowl_bool].copy()

#     # Calculate super_bowl_favorite_won_by_mean
#     favored_team_in_superbowl_bool = games_with_superbowl_favorite.loc[:,'favored_team_in_superbowl'] == 1
#     favored_sum = games_with_superbowl_favorite.loc[favored_team_in_superbowl_bool,'favorite_won_by'].sum()
#     unfavored_sum = games_with_superbowl_favorite.loc[~favored_team_in_superbowl_bool,'favorite_won_by'].sum()
#     super_bowl_favorite_won_by_mean = (favored_sum - unfavored_sum) / games_with_superbowl_favorite.shape[0]
    
#     # Calculate super_bowl_favorite_consecutive_wins
#     favored_team_in_superbowl_all = all_games_with_superbowl_favorite.loc[:,'favored_team_in_superbowl'] == 1
#     all_games_with_superbowl_favorite.loc[:,'superbowl_favorite_won'] = ((all_games_with_superbowl_favorite.loc[:,'favorite_won_by'] > 0) & favored_team_in_superbowl_all) | ((all_games_with_superbowl_favorite.loc[:,'favorite_won_by'] < 0) & ~favored_team_in_superbowl_all)
#     s = all_games_with_superbowl_favorite.loc[:,'superbowl_favorite_won']
#     super_bowl_favorite_consecutive_wins = np.array((~s).cumsum()[s].value_counts(sort=False))[-1]
    
#     # Calculate points_against_favorite & points_for_favorite
#     favored_team_in_superbowl_hometeam = all_games_with_superbowl_favorite.loc[:,'team_home'] == superbowl_favorite
#     points_against_favorite = (all_games_with_superbowl_favorite.loc[favored_team_in_superbowl_hometeam,'score_away'].sum() + 
#                                all_games_with_superbowl_favorite.loc[~favored_team_in_superbowl_hometeam,'score_home'].sum())
#     points_for_favorite = (all_games_with_superbowl_favorite.loc[favored_team_in_superbowl_hometeam,'score_home'].sum() + 
#                            all_games_with_superbowl_favorite.loc[~favored_team_in_superbowl_hometeam,'score_away'].sum())
#     # Calculate wins_for_favorite
#      home_team_wins = all_games_with_superbowl_favorite.loc[:,'score_home'] > all_games_with_superbowl_favorite.loc[:,'score_away']
#      wins_for_favorite = (all_games_with_superbowl_favorite.loc[favored_team_in_superbowl_hometeam & home_team_wins,'score_home'].size + 
#                           all_games_with_superbowl_favorite.loc[~favored_team_in_superbowl_hometeam & ~home_team_wins,'score_away'].size)
    
#     # Calculate points_against_unfavorite & points_for_unfavorite
#     unfavored_team_in_superbowl_hometeam = all_games_with_superbowl_unfavorite.loc[:,'team_home'] == superbowl_unfavorite
#     points_against_unfavorite = (all_games_with_superbowl_unfavorite.loc[unfavored_team_in_superbowl_hometeam,'score_away'].sum() + 
#                                  all_games_with_superbowl_unfavorite.loc[~unfavored_team_in_superbowl_hometeam,'score_home'].sum())
#     points_for_unfavorite = (all_games_with_superbowl_unfavorite.loc[unfavored_team_in_superbowl_hometeam,'score_home'].sum() + 
#                                  all_games_with_superbowl_unfavorite.loc[~unfavored_team_in_superbowl_hometeam,'score_away'].sum())
#     # Calculate wins_for_unfavorite
#     home_team_wins = all_games_with_superbowl_unfavorite.loc[:,'score_home'] > all_games_with_superbowl_unfavorite.loc[:,'score_away']
#     wins_for_unfavorite = (all_games_with_superbowl_unfavorite.loc[unfavored_team_in_superbowl_hometeam & home_team_wins,'score_home'].size + 
#                          all_games_with_superbowl_unfavorite.loc[~unfavored_team_in_superbowl_hometeam & ~home_team_wins,'score_away'].size)
    
#     # Calculate super_bowl_unfavorite_won_by_mean
#     favored_team_in_superbowl = games_with_superbowl_unfavorite.loc[:,'favored_team_in_superbowl'] == 1
#     favored_sum = games_with_superbowl_unfavorite.loc[favored_team_in_superbowl,'favorite_won_by'].sum()
#     unfavored_sum = games_with_superbowl_unfavorite.loc[~favored_team_in_superbowl,'favorite_won_by'].sum()
#     super_bowl_unfavorite_won_by_mean = (favored_sum - unfavored_sum) / games_with_superbowl_unfavorite.shape[0]
#     #print(super_bowl_unfavorite_won_by_mean)
    
#     # Calculate super_bowl_unfavorite_consecutive_wins
#     favored_team_in_superbowl_all = all_games_with_superbowl_unfavorite.loc[:,'favored_team_in_superbowl'] == 1
#     all_games_with_superbowl_unfavorite.loc[:,'superbowl_unfavorite_won'] = (((all_games_with_superbowl_unfavorite.loc[:,'favorite_won_by'] > 0) & favored_team_in_superbowl_all) | 
#                                                                              ((all_games_with_superbowl_unfavorite.loc[:,'favorite_won_by'] < 0)  & ~favored_team_in_superbowl_all))
#     s = all_games_with_superbowl_unfavorite.loc[:,'superbowl_unfavorite_won']
#     super_bowl_unfavorite_consecutive_wins = np.array((~s).cumsum()[s].value_counts(sort=False))[-1]

#     # calculate super_bowl_favorite_won_by_mean
#     favored_team_in_superbowl = games_with_superbowl_favorite.loc[:,'favored_team_in_superbowl'] == 1
#     favored_sum = games_with_superbowl_favorite.loc[favored_team_in_superbowl,'favorite_beat_spread_by'].sum()
#     unfavored_sum = games_with_superbowl_favorite.loc[~favored_team_in_superbowl,'favorite_beat_spread_by'].sum()
#     super_bowl_favorite_beat_spread_by_mean = (favored_sum - unfavored_sum) / games_with_superbowl_favorite.shape[0]
#     #print(super_bowl_favorite_won_by_mean)
    
#     # calculate super_bowl_unfavorite_won_by_mean
#     favored_team_in_superbowl = games_with_superbowl_unfavorite.loc[:,'favored_team_in_superbowl'] == 1
#     favored_sum = games_with_superbowl_unfavorite.loc[favored_team_in_superbowl,'favorite_beat_spread_by'].sum()
#     unfavored_sum = games_with_superbowl_unfavorite.loc[~favored_team_in_superbowl,'favorite_beat_spread_by'].sum()
#     super_bowl_unfavorite_beat_spread_by_mean = (favored_sum - unfavored_sum) / games_with_superbowl_unfavorite.shape[0]
#     #print(super_bowl_unfavorite_won_by_mean)
    
#     is_superbowl_in_year = (superbowls_since_1980.loc[:,'schedule_season'] == year)
    
#     superbowls_since_1980.loc[is_superbowl_in_year,'super_bowl_favorite_beat_spread_by_mean'] = super_bowl_favorite_beat_spread_by_mean
#     superbowls_since_1980.loc[is_superbowl_in_year,'super_bowl_unfavorite_beat_spread_by_mean'] = super_bowl_unfavorite_beat_spread_by_mean
#     superbowls_since_1980.loc[is_superbowl_in_year,'delta'] = superbowls_since_1980.loc[is_superbowl_in_year,'super_bowl_favorite_beat_spread_by_mean'] - superbowls_since_1980.loc[is_superbowl_in_year,'super_bowl_unfavorite_beat_spread_by_mean']
#     superbowls_since_1980.loc[is_superbowl_in_year,'super_bowl_favorite_consecutive_wins'] = super_bowl_favorite_consecutive_wins
#     superbowls_since_1980.loc[is_superbowl_in_year,'super_bowl_unfavorite_consecutive_wins'] = super_bowl_unfavorite_consecutive_wins
#     superbowls_since_1980.loc[:,'over_under_line'] = pd.to_numeric(superbowls_since_1980.loc[:,'over_under_line'])
    
#     superbowls_since_1980.loc[is_superbowl_in_year,'points_against_favorite'] = points_against_favorite.copy()
#     superbowls_since_1980.loc[is_superbowl_in_year,'points_for_unfavorite'] = points_for_unfavorite.copy()
#     superbowls_since_1980.loc[is_superbowl_in_year,'wins_for_favorite'] = wins_for_favorite
#     superbowls_since_1980.loc[is_superbowl_in_year,'wins_for_unfavorite'] = wins_for_unfavorite
#     superbowls_since_1980.loc[is_superbowl_in_year,'points_for_favorite'] = points_for_favorite.copy()
#     superbowls_since_1980.loc[is_superbowl_in_year,'points_against_unfavorite'] = points_against_unfavorite.copy()

# superbowls_since_1980.loc[:,'favorite_coach_win_%'] = [0.4,0.75,0.59,0.875,0.86,0.75,0.5,0.57,0.69,0.75,0.6,0.78,0.75,0.71,0.72,0.75,0.67,0.75,0.86,0.56,0.75,0.67,0.75,0.86,0.9,0.55,0.5,0.83,0.66,0.75,0.66,0.74,0.75,0.62,0.7,0.6,0.71,0.74]
# superbowls_since_1980.loc[:,'favorite_offense_rank'] = [6,7,10,1,2,2,8,4,7,1,1,1,2,2,1,3,1,2,2,1,14,1,2,12,4,9,2,1,20,7,10,3,11,1,4,1,3,2,]
# superbowls_since_1980.loc[:,'favorite_defense_rank'] = [1,2,2,11,1,1,2,7,8,3,6,2,5,2,6,3,1,5,8,4,1,7,6,1,2,3,23,4,1,8,2,15,2,22,8,6,1,5,]
# superbowls_since_1980.loc[:,'unfavorite_offense_rank'] = [7,3,12,3,1,10,6,4,1,8,15,2,3,7,5,5,2,1,4,7,15,6,18,15,8,1,2,14,3,1,12,9,10,8,10,19,1,3,]
# superbowls_since_1980.loc[:,'unfavorite_defense_rank'] =  [10,12,1,13,7,6,15,6,16,1,1,19,14,5,9,9,14,6,4,15,5,6,1,10,2,7,3,17,28,20,1,25,12,1,1,4,27,4]
# superbowls_since_1980.loc[:,'favorite_turnovers'] = [28,25,23,18,22,31,32,36,26,25,21,23,24,22,24,23,24,32,20,31,26,44,19,24,27,23,19,15,25,24,22,17,16,26,13,19,11,12,]
# superbowls_since_1980.loc[:,'favorite_opp_turnovers'] = [35,48,27,61,38,54,43,47,38,37,35,41,31,28,35,25,39,32,30,36,49,34,31,41,36,30,26,31,29,26,32,34,25,26,25,39,23,18,]
# superbowls_since_1980.loc[:,'unfavorite_turnovers'] = [44,24,16,49,28,42,29,37,27,32,14,35,38,35,23,34,27,21,24,22,24,28,21,31,22,17,36,34,30,28,18,24,16,19,14,31,11,20,]
# superbowls_since_1980.loc[:,'unfavorite_opp_turnovers'] = [52,37,24,36,36,47,35,34,36,43,34,37,35,47,32,34,34,31,44,40,31,35,38,26,28,27,44,25,30,39,35,31,25,39,24,27,22,31,]
# superbowls_since_1980.loc[:,'temp'] = [65,65,54,51,45,65,56,58,72,65,62,65,56,65,73,54,65,59,74,65,65,65,81,65,59,65,67,65,66,60,65,65,65,49,65,76,65,65]
# superbowls_since_1980.loc[:,'wind'] = [0,0,9,16,8,0,7,6,17,0,9,0,10,0,6,5,0,6,11,0,8,0,0,0,12,0,10,0,1,6,0,0,0,4,0,16,0,0]
