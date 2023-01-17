
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.express as px
import plotly.graph_objects as go

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/brazilian-football-championship/BRA.csv')

data.head()
data.drop(data.iloc[:, 10:19], inplace = True, axis = 1)

data.head()
def update_vals(row, data=data):
    if row.Home == 'Chapecoense-SC':
        if row.Date == '11/12/2016':
            row.Res = 'A'
            row.HG = 0
            row.AG = 3
    return row

data = data.apply(update_vals, axis=1)
data = data.append(pd.Series(['Brazil','Serie A', 2012, '11/12/2016', '22:30', 'Chapecoense-SC', 'Atletico-MG', 0, 3, 'H'], index=data.columns), ignore_index=True)
season = data.groupby(by="Season").sum().sort_values(by="Season", ascending=False).reset_index()

season
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=season['Season'],
    y=season['HG'],
    name="Home Goals",
))


fig.add_trace(go.Scatter(
    x=season['Season'],
    y=season['AG'],
    name="Away Goals"
))

fig.update_layout(
    title="Home and Away goals by Season",
    xaxis_title="Season",
    yaxis_title="Goals",
    legend_title="Type",
    font=dict(
        size=18,
    )
)

fig.show()
teams = data.groupby(by="Home").sum().sort_values(by="Season", ascending=False).reset_index()

teams
fig = px.scatter(teams, x="HG", y="AG", color=teams['Home'])

fig.update_layout(
    title="Home and Away goals by Club",
    xaxis_title="Conceded Goals",
    yaxis_title="Scored Goals",
    legend_title="Team",
    font=dict(
        size=18,
    ),
)

fig.show()
home_results = data.groupby(by="Home")

home_results_teams = pd.DataFrame(columns=["Team", "Win", "Draw", "Lose"])

rows = []
for index, item in enumerate(home_results):
    home_results_teams.loc[index, 'Team'] = item[0]
    temp_df = item[1].groupby(by="Res").count()
    temp_df = temp_df['Home'].reset_index()

    home_results_teams.loc[index, 'Win'] = temp_df['Home'][2]
    home_results_teams.loc[index, 'Draw'] = temp_df['Home'][1]
    home_results_teams.loc[index, 'Lose'] = temp_df['Home'][0]

    
home_results_teams = home_results_teams.sort_values(by="Win", ascending=False)

home_results_teams.reset_index(drop=True)
fig = go.Figure(data=[
    go.Bar(name='Win', x=home_results_teams['Team'], y=home_results_teams['Win']),
    go.Bar(name='Draw', x=home_results_teams['Team'], y=home_results_teams['Draw']),
    go.Bar(name='Lose', x=home_results_teams['Team'], y=home_results_teams['Lose'])
])
fig.update_layout(
    barmode='group',
    title="Results by Club playing as home",
    legend_title="Result",
)
fig.show()
away_results = data.groupby(by="Away")

away_results_teams = pd.DataFrame(columns=["Team", "Win", "Draw", "Lose"])

rows = []
for index, item in enumerate(away_results):
    away_results_teams.loc[index, 'Team'] = item[0]
    temp_away_df = item[1].groupby(by="Res").count()
    temp_away_df = temp_away_df['Away'].reset_index()

    try:
        away_results_teams.loc[index, 'Win'] = temp_away_df['Away'][0]
        away_results_teams.loc[index, 'Draw'] = temp_away_df['Away'][1]
        away_results_teams.loc[index, 'Lose'] = temp_away_df['Away'][2]
    except:
        print('ERR')

    
away_results_teams = away_results_teams.sort_values(by="Win", ascending=False)

away_results_teams.reset_index(drop=True)
fig = go.Figure(data=[
    go.Bar(name='Win', x=away_results_teams['Team'], y=away_results_teams['Win']),
    go.Bar(name='Draw', x=away_results_teams['Team'], y=away_results_teams['Draw']),
    go.Bar(name='Lose', x=away_results_teams['Team'], y=away_results_teams['Lose'])
])
fig.update_layout(
    barmode='group',
    title="Results by Club playing as away",
    legend_title="Result",
)
fig.show()