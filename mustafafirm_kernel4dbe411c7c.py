# Import libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson,skellam
# Import Home/Away Dataset

home_away = pd.read_csv("../input/home_away.csv", index_col = 'Rk')
home_away
# Import fixture_list data
fixtures = pd.read_csv('../input/Fix.csv')
fixtures.head(2)
home_away = home_away.copy()
# Rename Columns to sensible names
home_away.rename(columns = {'MP': 'MP (H)', 'MP.1': 'MP (A)', 'xGDiff/90': 'xGDiff/90 (H)', 
                     'xGDiff/90.1': 'xGDiff/90 (A)', 'Pts': 'Pts (H)', 'Pts.1': 'Pts (A)', 
                        'xG': 'xG (H)', 'xG.1': 'xG (A)', 'xGA': 'xGA (H)', 'xGA.1': 'xGA (A)'
                           } , inplace = True)
# Cleaning team names
home_away.replace(to_replace = ['KÃ¶ln', 'DÃ¼sseldorf'] , value = ['Koln', 'Dusseldorf'], inplace = True)
# Extracting needed columns
home_away = home_away[['Squad', 'MP (H)', 'MP (A)', 'xGDiff/90 (H)', 'xGDiff/90 (A)', 'Pts (H)',
                       'Pts (A)', 'xG (H)', 'xG (A)', 'xGA (H)', 'xGA (A)']]
# Getting total points as a feature
home_away['Pts'] = home_away['Pts (H)'] + home_away['Pts (A)']
home_away
# Current league standings
standings = home_away[['Squad', 'Pts']]
standings.head(2)
home_away['Home_adv_factor'] = home_away['xGDiff/90 (H)'] - home_away['xGDiff/90 (A)']
mode = home_away[['Squad', 'Home_adv_factor']]
mode.plot(kind = 'bar' , figsize = (15,10), x = 'Squad', y = 'Home_adv_factor')

home_away[['Squad', 'xGA (H)']].sort_values('xGA (H)').head()
home_away[['Squad', 'xGA (A)']].sort_values('xGA (A)').head(8)
home_away[['Squad', 'Pts (H)']].sort_values('Pts (H)').head(3)
# Recall info about the dataset
fixtures.info()
# Cleaning column names
fixtures.rename(columns = {'xG': 'xG (H)', 'xG.1': 'xG (A)'}, inplace = True)
# Cleaning team names
fixtures.replace(to_replace = ['KÃ¶ln', 'DÃ¼sseldorf'] , value = ['Koln', 'Dusseldorf'], inplace = True)
# Removing rows containing absloutely no information
fixtures.dropna(how = 'all', inplace = True)
# Extract dataset for the remaining_fixtures leaving the fixtures dataset containing games already played
remaining_fixtures = fixtures[fixtures['xG (H)'].isnull()] 
remaining_fixtures = remaining_fixtures[['Home', 'Away']].reset_index()
del remaining_fixtures['index']
remaining_fixtures.head(2)
# Extracting needed features and cleaning dataset
fixtures = fixtures.copy()
fixtures = fixtures[['Home', 'Away', 'xG (H)', 'xG (A)']]
# Removing fixtures still to be played.
fixtures.dropna(how='any', inplace = True, axis = 'rows')

# Rounding expected goals scored and conceaded for each team.
fixtures['xG_rounded (H)'] = fixtures['xG (H)'].apply(lambda x: round(x))
fixtures['xG_rounded (A)'] = fixtures['xG (A)'].apply(lambda x: round(x))
fixtures = fixtures[['Home', 'Away', 'xG_rounded (H)', 'xG_rounded (A)']]
# Simulating game results based on expected goals.
fixtures['xwin'] = fixtures.apply(lambda x: 1 if x['xG_rounded (H)'] > x['xG_rounded (A)'] else 0, axis=1)
fixtures['xdraw'] = fixtures.apply(lambda x: 1 if x['xG_rounded (H)'] == x['xG_rounded (A)'] else 0, axis=1)
fixtures['xloss'] = fixtures.apply(lambda x: 1 if x['xG_rounded (H)'] < x['xG_rounded (A)'] else 0, axis=1)
squad_list = list(home_away['Squad'].values)
# xG Position
# Get the results for each team
x_team_results = []
for team in squad_list:
    # Get the data for that team
    team_data_home = fixtures[fixtures['Home'] == team]
    team_data_away = fixtures[fixtures['Home'] == team]
    
    
    wins = team_data_home['xwin'].sum()
    draws = team_data_home['xdraw'].sum()
    losses = team_data_home['xloss'].sum()
    scored = team_data_home['xG_rounded (H)'].sum()
    conceded = team_data_home['xG_rounded (A)'].sum()
    
    wins_a = team_data_away['xwin'].sum()
    draws_a = team_data_away['xdraw'].sum()
    losses_a = team_data_away['xloss'].sum()
    scored_a = team_data_away['xG_rounded (H)'].sum()
    conceded_a = team_data_away['xG_rounded (A)'].sum()
    
    games = wins + draws + losses + wins_a + draws_a + losses_a
    points = (3 * wins) + draws + (3 * wins_a) + draws_a
    goal_difference = (scored - conceded) + (scored_a - conceded_a)
    
    x_team_results.append([team, games, wins+wins_a , draws + draws_a , losses + losses_a
                           , scored + scored_a, conceded + conceded_a, 
                           goal_difference, points])
x_league_table = pd.DataFrame(x_team_results, 
                              columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Points'])
x_league_table.sort_values(by=['Points', 'GD', 'F'], ascending=False, inplace=True)
x_league_table.set_index(np.array(range(1, 19)), inplace=True)
x_league_table.index.names = ['xG_Position']
x_league_table
team_positions = []
for team in squad_list:
    current_pos = standings[standings['Squad'] == team].index[0]
    xg_pos = x_league_table[x_league_table['Team'] == team].index[0]
    overperforming = 'Yes' if current_pos < xg_pos else 'No'
    team_positions.append([team, current_pos, xg_pos,  overperforming])

position_table = pd.DataFrame(team_positions, columns=['Team', 'Position', 'xG Position',
                                                       'Overperforming'])
position_table.sort_values(by=['Position'], ascending=True, inplace=True)
position_table.head(18)
# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

def model(goal_model_data, home_adv):
    if home_adv == 1:
        poisson_model = smf.glm(formula="goals ~  home +  team + opponent  ", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
        print(poisson_model.summary())
    else:
        poisson_model = smf.glm(formula="goals ~  team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
        print(poisson_model.summary())
    return poisson_model
def simulate_match(foot_model, homeTeam, awayTeam, max_goals , home_adv):
    if home_adv == 1:
        home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
        away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    else:
        home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam, 'home': 0},
                                                      index=[1])).values[0]
        away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam, 'home': 0},
                                                      index=[1])).values[0]
            
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
df_xG = pd.concat([fixtures[['Home','Away','xG_rounded (H)']].assign(home=1).rename(
            columns={'Home':'team', 'Away':'opponent','xG_rounded (H)':'goals'}),
                             fixtures[['Away','Home','xG_rounded (A)']].assign(home=0).rename(
            columns={'Away':'team', 'Home':'opponent','xG_rounded (A)':'goals'})], sort = False)
df_xG.tail(2)
# Method to compute rank for each team while simulating its league campaign
def compute_rank(data, home_adv = 1, r = remaining_fixtures, home_away = home_away):
    winners = []
    poisson_model = model(data, home_adv)
    for home,away in r.values:
        avg_no_of_goals = poisson_model.predict(pd.DataFrame(data={'team': home, 'opponent': away,
                                       'home':1},index=[1]))
        to_fn = int((avg_no_of_goals.values[0]))
        match_sim = simulate_match(poisson_model, home, away, to_fn, home_adv)
        result = np.argmax(np.array([np.sum(np.tril(match_sim, -1)), np.sum(np.diag(match_sim)),
                                np.sum(np.triu(match_sim, 1))]))
    
        if result == 0 :
                winner = home
        elif result == 1:
                winner = 'None'
        else:
                winner = away

        winners.append(winner)
    winners = pd.DataFrame(winners)
    r = pd.concat([r, winners], axis = 1)
    r.columns = ['Home', 'Away', 'Winner']
    wins_pts = pd.DataFrame((r['Winner'].value_counts()) *3)
    wins_pts = wins_pts.drop('None')
    draws = r[r['Winner'] == 'None']
    draws_pts_h = pd.DataFrame(draws['Home'].value_counts())
    draws_pts_a = pd.DataFrame(draws['Away'].value_counts())
    Total = pd.concat([wins_pts, draws_pts_h, draws_pts_a], axis = 1, sort = True)
    Total.fillna(0, inplace = True)
    Total['Total_pts_to_be_Added'] = (Total.Winner + Total.Home + Total.Away).astype(int)
    Total.sort_values('Total_pts_to_be_Added', ascending = False)
    home_away = home_away.set_index('Squad')
    Results = pd.concat([home_away['Pts'], Total], axis = 1, sort = False)
    Results = Results.fillna(0)
    Ranking = pd.DataFrame(Results['Pts'] + Results['Total_pts_to_be_Added'])
    Ranking.columns = ['Points']
    Ranking = Ranking.astype({'Points': int})
    Ranking.sort_values('Points', ascending = False, inplace = True)
    Ranking.reset_index(inplace = True)
    Ranking.index.name = 'Predicted Position'
    Ranking.index += 1
    Ranking.rename( columns = {'index': 'Teams'}, inplace = True)
    return Ranking
# League Table Ranking predicted through xG with Home Advantage for each team
xG_rankings_with_home_adv = compute_rank(df_xG)
xG_rankings_with_home_adv
# League Table Ranking predicted through xG without the home advantage
xG_rankings_without_home_adv = compute_rank(df_xG,0)
xG_rankings_without_home_adv

pos = []
for team in squad_list:
    home_pos = xG_rankings_with_home_adv[xG_rankings_with_home_adv['Teams'] == team].index[0]
    no_home_pos = xG_rankings_without_home_adv[xG_rankings_without_home_adv['Teams'] == team].index[0]
    affected = 'Yes' if home_pos != no_home_pos else 'No'
    pos.append([team, home_pos, no_home_pos, affected])
    
cmp_table = pd.DataFrame(pos, columns = ['Team', 'Position_with_home_adv', 'Position_without_home_adv', 'affected']) 
cmp_table.set_index('Team', inplace = True)
cmp_table.sort_values('Position_with_home_adv')
