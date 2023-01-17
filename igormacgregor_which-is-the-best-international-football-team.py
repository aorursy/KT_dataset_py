import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import plotly.express as px

from datetime import datetime
code_df = pd.read_csv('../input/country-code/country_code.csv',  usecols=['Country_name', 'code_3digit'])
code_df.head()
games = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
games.isnull().sum().sum()
#Nice
games['Winner'] = np.where((games['home_score'] > games['away_score']), games['home_team'], np.where((games['home_score'] < games['away_score']), games['away_team'], 'Draw'))
games['Loser'] = np.where((games['home_score'] > games['away_score']), games['away_team'], np.where((games['home_score'] < games['away_score']), games['home_team'], 'Draw'))

games['Year'] = games['date'].apply(lambda date : int(date.split('-')[0]))
games['Month'] = games['date'].apply(lambda date : int(date.split('-')[1]))
games['Day'] = games['date'].apply(lambda date : int(date.split('-')[2]))
games['Date'] = games['date'].apply(lambda date : pd.to_datetime(date))
games.head()
fifa_ranking = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
fifa_ranking = fifa_ranking[fifa_ranking['rank_date'] >= '2011-08-01']
ranking_df = pd.pivot_table(data = fifa_ranking, 
                            values = 'total_points',
                            index = 'country_full',
                            columns = 'rank_date').fillna(0.0)
ranking_df.head()
import plotly.express as px
best_ranks = ranking_df.loc[np.append(ranking_df.idxmax().unique(),'France')]
fig = go.Figure()

for i in range(len(best_ranks.values)):
    fig.add_trace(go.Scatter(x = best_ranks.columns, 
                             y = best_ranks.iloc[i],
                             name = best_ranks.index[i]))
fig.show()
real_world_champion = games.iloc[1]['Winner'] # Game 0 is a draw
day_one = games.iloc[1]['Date'] # Game 0 is a draw
champions = [real_world_champion]
dates = [day_one]
champions_time = {}

for i in range(len(games)):
    if games.iloc[i].Loser == real_world_champion:
        if real_world_champion in champions_time:
            champions_time[real_world_champion] += (games.loc[i, 'Date'] - dates[-1]).days
        else:
            champions_time[real_world_champion] = (games.loc[i, 'Date'] - dates[-1]).days
        real_world_champion = games.loc[i, 'Winner']
        champions.append(real_world_champion)
        dates.append(games.loc[i, 'Date'])
champions_time[real_world_champion] += (datetime.now() - dates[-1]).days
countries_df = pd.DataFrame.from_dict(champions_time, orient = 'index', columns=['Days champion'])

print(champions[0], real_world_champion)
def return_country_code(con):
    if con in code_df['Country_name'].values:
        return code_df[code_df['Country_name'] == con]['code_3digit'].values[0]
    elif con == 'United States':
        return code_df[code_df['Country_name'] == 'United States of America']['code_3digit'].values[0]
    elif con == 'Russia': 
        return code_df[code_df['Country_name'] == 'Russian Federation']['code_3digit'].values[0]
    elif con == 'South Korea': 
        return code_df[code_df['Country_name'] == 'Korea (South)']['code_3digit'].values[0]
    elif con == 'Republic of Ireland': 
        return code_df[code_df['Country_name'] == 'Ireland']['code_3digit'].values[0]
    elif con == 'North Korea': 
        return code_df[code_df['Country_name'] == 'Korea (North)']['code_3digit'].values[0]
    elif con == 'Venezuela': 
        return code_df[code_df['Country_name'] == 'Venezuela (Bolivarian Republic)']['code_3digit'].values[0]
    elif con == 'China PR': 
        return code_df[code_df['Country_name'] == 'China']['code_3digit'].values[0]

countries_df['Country'] = countries_df.index
countries_df['Code'] = countries_df['Country'].apply(return_country_code)
uk_df = pd.DataFrame([[(countries_df.loc['England']['Days champion'] +  countries_df.loc['Scotland']['Days champion'] + countries_df.loc['Northern Ireland']['Days champion'] + countries_df.loc['Wales']['Days champion']) / 4,
                          'United Kingdom',
                           code_df[code_df['Country_name'] == 'United Kingdom']['code_3digit'].values[0]
                         ]], 
                         index = ['United Kingdom'], 
                         columns = ['Days champion', "Country", 'Code'])
final_uc_df = countries_df.append(uk_df).dropna().sort_values('Days champion', ascending = False)
final_uc_df.head()
data=dict(
    type = 'choropleth',
    locations = final_uc_df['Code'],
    z = final_uc_df['Days champion'],
    text = final_uc_df['Country'],
    colorscale = 'YlOrRd',
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Number of days being unofficial Champions',
)

layout = dict(title_text='The Longest Unofficial Champions',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ))

fig = go.Figure(data = [data], layout = layout)
iplot(fig)
top_uc_df = countries_df.sort_values('Days champion', ascending = False)[:15]
plt.figure(figsize = (14,4))
sns.barplot(x=top_uc_df['Country'], y=top_uc_df['Days champion'], palette="vlag")
df_timeline = []
for i in range(len(champions) - 1):
    df_timeline.append(dict(Task = champions[i], Start=dates[i], Finish=dates[i + 1]))
df_timeline.append(dict(Task = champions[-1], Start=dates[-1], Finish=datetime.now()))
fig = ff.create_gantt(df_timeline, group_tasks=True, title='Gantt Chart of the Unofficial World Champion')
fig.show()
games[(games['Winner'] == 'Belarus') & (games['date'] >= '1967-06-04') & (games['date'] <= '1979-07-25')]
def make_stats_df(df):
    my_columns =  ['Wins', 'Draws', 'Loses', 'Total games', 'Goals scored', 'Goals taken', 
                   'Goals difference', 'World Cup games', 'World Cup wins']
    data_df = pd.DataFrame(0, index = df['home_team'].append(df['away_team']).unique(), columns = my_columns)
    
    for i in range(len(df)):
        if df.iloc[i]['Winner'] == 'Draw':
            data_df.loc[df.iloc[i]['home_team']]['Draws'] += 1
            data_df.loc[df.iloc[i]['away_team']]['Draws'] += 1
        else:
            data_df.loc[df.iloc[i]['Winner']]['Wins'] += 1
            data_df.loc[df.iloc[i]['Loser']]['Loses'] += 1
        if df.iloc[i]['tournament'] == 'FIFA World Cup':
            data_df.loc[df.iloc[i]['home_team']]['World Cup games'] += 1
            data_df.loc[df.iloc[i]['away_team']]['World Cup games'] += 1
            if df.iloc[i]['Winner'] != 'Draw':
                data_df.loc[df.iloc[i]['Winner']]['World Cup wins'] += 1
        data_df.loc[df.iloc[i]['home_team']]['Goals scored'] += df.iloc[i]['home_score']
        data_df.loc[df.iloc[i]['home_team']]['Goals taken'] += df.iloc[i]['away_score']
        data_df.loc[df.iloc[i]['away_team']]['Goals scored'] += df.iloc[i]['away_score']
        data_df.loc[df.iloc[i]['away_team']]['Goals taken'] += df.iloc[i]['home_score']

    data_df['Total games'] = data_df['Wins'] + data_df['Draws'] + data_df['Loses']
    data_df['Goals difference'] = data_df['Goals scored'] - data_df['Goals taken']
    data_df['Winning rate'] = data_df['Wins'] / data_df['Total games']
    data_df['Goals per game'] = data_df['Goals scored'] / data_df['Total games']
    data_df['Average difference'] = data_df['Goals difference'] / data_df['Total games']
    data_df['WC Winning rate'] = data_df['World Cup wins'] / data_df['World Cup games']
    return data_df
data_df = make_stats_df(games)

plt.figure(figsize = (12,8))
plt.subplot(311)
plt.title('International Teams by Winning Rate')
sns.barplot(x = data_df[data_df['Total games'] >= 100].sort_values('Winning rate', ascending = False).head(10).index, y = data_df[data_df['Total games'] >= 50].sort_values('Winning rate', ascending = False).head(10)['Winning rate'], palette="vlag")
plt.subplot(312)
plt.title('International Teams by Total Average Difference')
sns.barplot(x = data_df[data_df['Total games'] >= 100].sort_values('Average difference', ascending = False).head(10).index, y = data_df[data_df['Total games'] >= 50].sort_values('Average difference', ascending = False).head(10)['Average difference'], palette="vlag")
plt.subplot(313)
plt.title('International Teams by Winning Rate at the FIFA World Cup')
sns.barplot(x = data_df[data_df['World Cup games'] >= 10].sort_values('WC Winning rate', ascending = False).head(10).index, y = data_df[data_df['World Cup games'] >= 10].sort_values('WC Winning rate', ascending = False).head(10)['WC Winning rate'], palette="vlag")
plt.tight_layout()
modern_data = make_stats_df(games[(games['Date'] >= pd.to_datetime('07-07-1957')) & (games['tournament'] != 'Friendly')])

plt.figure(figsize = (12,8))
plt.subplot(211)
plt.title('International Teams by Winning Rate since 1957')
sns.barplot(x = modern_data[modern_data['Total games'] >= 100].sort_values('Winning rate', ascending = False).head(10).index, y = modern_data[modern_data['Total games'] >= 50].sort_values('Winning rate', ascending = False).head(10)['Winning rate'], palette="vlag")
plt.subplot(212)
plt.title('International Teams by Goal Average since 1957')
sns.barplot(x = modern_data[modern_data['Total games'] >= 100].sort_values('Average difference', ascending = False).head(10).index, y = modern_data[modern_data['Total games'] >= 50].sort_values('Average difference', ascending = False).head(10)['Average difference'], palette="vlag")
all_teams = games['home_team'].append(games['away_team']).unique()
elo_df = pd.DataFrame(0, index = all_teams, columns= range(1870,2020))

major_comp = ['UEFA Euro', 'African Cup of Nations', 'Copa América', 'AFC Asian Cup', 'UEFA Nations League',
              'Confederations Cup', 'African Nations Championship', 'CONCACAF Championship', 'Gold Cup',
             'Pan American Championship', 'Pacific Games', 'Oceania Nations Cup']
qualif = ['Copa América qualification', 'AFC Asian Cup qualification', 'UEFA Euro qualification', 
          'African Cup of Nations qualification', 'FIFA World Cup qualification', 'CONCACAF Championship qualification',
          'Gold Cup qualification', 'Oceania Nations Cup qualification']

def def_k(comp):
    if comp == 'FIFA World Cup':
        return 60
    elif comp in major_comp:
        return 50
    elif comp in qualif:
        return 40
    elif comp == 'Friendly':
        return 20
    else:
        return 30

def def_g(team_goals, enemy_goals):
    if team_goals - enemy_goals <= 1:
        return 1
    elif team_goals - enemy_goals == 2:
        return 3/2
    elif team_goals - enemy_goals == 3:
        return 7/4
    else:
        return 7/4 + (team_goals - enemy_goals - 3)/8

def def_w(team, winner):
    if team == winner:
        return 1
    elif winner == 'Draw':
        return 1/2
    else:
        return 0
    
def def_dr(team_elo, enemy_elo, neutral):
    if neutral:
        return team_elo - enemy_elo
    else: 
        return team_elo - enemy_elo + 100

for year in range(1871, 2020):
    elo_df[year] = elo_df[year - 1]
    for game in games[games['Year'] == year].values:
        game_series = pd.Series(game, index = games.columns)
        elo_df.loc[game_series['home_team'], year] += def_k(game_series['tournament']) * def_g(game_series['home_score'], game_series['away_score']) * (def_w(game_series['home_team'], game_series['Winner']) - 1/(10 **(- def_dr(elo_df.loc[game_series['home_team']][year], elo_df.loc[game_series['away_team']][year], game_series['neutral']) / 400) + 1))
        elo_df.loc[game_series['away_team'], year] += def_k(game_series['tournament']) * def_g(game_series['away_score'], game_series['home_score']) * (def_w(game_series['away_team'], game_series['Winner']) - 1/(10 **(- def_dr(elo_df.loc[game_series['away_team']][year], elo_df.loc[game_series['home_team']][year], True) / 400) + 1))
plt.figure(figsize=(12,4))
plt.title('Years being #1 ELO team')
plt.bar(x = elo_df.idxmax().value_counts().index, height = elo_df.idxmax().value_counts().values)
plt.tight_layout()
best_elos = elo_df.loc[elo_df.idxmax().unique()]
fig = go.Figure(layout = dict(title='ELO Ranking of the best international teams'))

for i in range(len(best_elos.values)):
    fig.add_trace(go.Scatter(x = best_elos.columns, 
                             y = best_elos.iloc[i],
                             name = best_elos.index[i]))
fig.show()
data_df[data_df['Total games'] >= 150].sort_values('Winning rate').head(10)
# The only win of San Marino, out of 163 games !
games[games['Winner'] == 'San Marino']
# The game with the highest number of goals
games[(games['home_score'] + games['away_score']) == (games['home_score'] + games['away_score']).max()]
all_teams = games['Loser'].unique() #Only teams which have lost a game
enemys_df = pd.DataFrame('', index = all_teams[1:], columns = ['Worst enemy'])
for country in all_teams[1:]: #We don't take 'Draw'
    enemys_df.loc[country]['Worst enemy'] = games[games['Loser'] == country]['Winner'].value_counts().index[0]
enemys_df['Worst enemy'].value_counts().head(10)
enemys_df.loc[best_elos.index]
#We exclude teams that haven't win a single game
win_streaks = pd.DataFrame('', index = games['Winner'].unique()[1:], columns = ['Longest streak', 'Start of the streak', 'End of the streak', 'End of streak opponent'])
for team in games['Winner'].unique()[1:]:
    team_games = games[(games['home_team'] == team) | (games['away_team'] == team)]
    team_games['won'] = (team_games['Winner'] == team).apply(int)
    team_games['series'] = (team_games['won'] != team_games['won'].shift()).cumsum()
    team_games['streak'] = team_games.groupby(['won', 'series']).cumcount() + 1
    team_games.loc[team_games['won'] == 0, 'streak'] = 0
    #Find longest streak
    win_streaks.loc[team, 'Longest streak'] = team_games['streak'].max()
    last_win = team_games.loc[team_games['streak'].idxmax()]
    win_streaks.loc[team, 'Start of the streak'] = team_games.loc[(team_games['series'] == last_win['series']) & (team_games['streak'] == 1),'Date'].values[0]
    if team_games.loc[(team_games['series'] == last_win['series'] + 1) & (team_games['streak'] == 0),'Date'].values.size == 0:
        win_streaks.loc[team, 'End of the streak'] = 'Currently on streak'
        win_streaks.loc[team, 'End of streak opponent'] = 'NA'
    else:
        win_streaks.loc[team, 'End of the streak'] = team_games.loc[(team_games['series'] == last_win['series'] + 1) & (team_games['streak'] == 0),'Date'].values[0]
        win_streaks.loc[team, 'End of streak opponent'] = team_games.loc[(team_games['series'] == last_win['series'] + 1) & (team_games['streak'] == 0),'Winner'].values[0]
win_streaks.sort_values(by= 'Longest streak', ascending = False).head(10)
invincibility_streaks = pd.DataFrame('', index = games['Winner'].unique()[1:], columns = ['Longest streak', 'Start of the streak', 'End of the streak', 'End of streak opponent'])
for team in games['Winner'].unique()[1:]:
    team_games = games[(games['home_team'] == team) | (games['away_team'] == team)]
    #Only the following condition changes
    team_games['won'] = (team_games['Loser'] != team).apply(int)
    team_games['series'] = (team_games['won'] != team_games['won'].shift()).cumsum()
    team_games['streak'] = team_games.groupby(['won', 'series']).cumcount() + 1
    team_games.loc[team_games['won'] == 0, 'streak'] = 0
    #Find longest streak
    invincibility_streaks.loc[team, 'Longest streak'] = team_games['streak'].max()
    last_win = team_games.loc[team_games['streak'].idxmax()]
    invincibility_streaks.loc[team, 'Start of the streak'] = team_games.loc[(team_games['series'] == last_win['series']) & (team_games['streak'] == 1),'Date'].values[0]
    if team_games.loc[(team_games['series'] == last_win['series'] + 1) & (team_games['streak'] == 0),'Date'].values.size == 0:
        invincibility_streaks.loc[team, 'End of the streak'] = 'Currently on streak'
        invincibility_streaks.loc[team, 'End of streak opponent'] = 'NA'
    else:
        invincibility_streaks.loc[team, 'End of the streak'] = team_games.loc[(team_games['series'] == last_win['series'] + 1) & (team_games['streak'] == 0),'Date'].values[0]
        invincibility_streaks.loc[team, 'End of streak opponent'] = team_games.loc[(team_games['series'] == last_win['series'] + 1) & (team_games['streak'] == 0),'Winner'].values[0]
invincibility_streaks.sort_values(by= 'Longest streak', ascending = False).head(10)
invincibility_streaks['Country'] = invincibility_streaks.index
invincibility_streaks['Code'] = invincibility_streaks['Country'].apply(return_country_code)
invincibility_streaks.head()
invincibility_streaks['text'] = invincibility_streaks['Start of the streak'].apply(pd.to_datetime).apply(str) + ' - ' + invincibility_streaks['End of the streak'].apply(str)
data=dict(
    type = 'choropleth',
    locations = invincibility_streaks['Code'],
    z = invincibility_streaks['Longest streak'],
    text = invincibility_streaks['text'],
    colorscale = 'YlOrRd',
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = '#games during longest streak',
)

layout = dict(title_text='The Longest Streaks of Invincibility per country',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ))

fig = go.Figure(data = [data], layout = layout)
iplot(fig)
