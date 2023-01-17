import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from os import path
from PIL import Image

# import plotly modules
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

%matplotlib inline
world_cups = pd.read_csv('../input/fifa-world-cup/WorldCups.csv')
world_cup_player = pd.read_csv('../input/fifa-world-cup/WorldCupPlayers.csv')
world_cups_matches = pd.read_csv('../input/fifa-world-cup/WorldCupMatches.csv')
#DROP NA VALUES
world_cup_player = world_cup_player.dropna()
world_cups = world_cups.dropna()
world_cups_matches = world_cups_matches.dropna()
world_cups = world_cups.replace('Germany FR', 'Germany')
world_cup_player = world_cup_player.replace('Germany FR', 'Germany')
world_cups_matches = world_cups_matches.replace('Germany FR', 'Germany')
world_cups['Attendance'] = world_cups['Attendance'].str.replace('.', '').astype('int64')
world_cups.head(2)
world_cup_player.head(2)
world_cups_matches.head(2)
gold = world_cups["Winner"]
silver = world_cups["Runners-Up"]
bronze = world_cups["Third"]

gold_count = pd.DataFrame.from_dict(gold.value_counts())
silver_count = pd.DataFrame.from_dict(silver.value_counts())
bronze_count = pd.DataFrame.from_dict(bronze.value_counts())
podium_count = gold_count.join(silver_count, how='outer').join(bronze_count, how='outer')
podium_count = podium_count.fillna(0)
podium_count.columns = ['WINNER', 'SECOND', 'THIRD']
podium_count = podium_count.astype('int64')
podium_count = podium_count.sort_values(by=['WINNER', 'SECOND', 'THIRD'], ascending=False)

podium_count.plot(y=['WINNER', 'SECOND', 'THIRD'], kind="bar", 
                  color =['gold','silver','brown'], figsize=(15, 6), fontsize=14,
                 width=0.8, align='center')
plt.xlabel('Countries')
plt.ylabel('Number of podium')
plt.title('Number of podium by country')
#world_cups_matches['Win conditions'].value_counts()
home = world_cups_matches[['Home Team Name', 'Home Team Goals']].dropna()
away = world_cups_matches[['Away Team Name', 'Away Team Goals']].dropna()

goal_per_country = pd.DataFrame(columns=['countries', 'goals'])
goal_per_country = goal_per_country.append(home.rename(index=str, columns={'Home Team Name': 'countries', 'Home Team Goals': 'goals'}))
goal_per_country = goal_per_country.append(away.rename(index=str, columns={'Away Team Name': 'countries', 'Away Team Goals': 'goals'}))

goal_per_country['goals'] = goal_per_country['goals'].astype('int64')

goal_per_country = goal_per_country.groupby(['countries'])['goals'].sum().sort_values(ascending=False)

goal_per_country[:10].plot(x=goal_per_country.index, y=goal_per_country.values, kind="bar", figsize=(12, 6), fontsize=14)
plt.xlabel('Countries')
plt.ylabel('Number of goals')
plt.title('Top 10 of Number of goals by country')
plt.figure(figsize = (22,12))
sns.set_style("whitegrid")
plt.subplot(221)
g1 = sns.barplot(x="Year", y="Attendance", data=world_cups, palette="Blues")
g1.set_title("ATTENDANCE PER CUP", fontsize=14)

plt.subplot(222)
g2 = sns.barplot(x="Year", y="QualifiedTeams", data=world_cups, palette="Blues")
g2.set_title("NUMBER OF TEAMS PER CUP", fontsize=14)

plt.subplot(223)
g2 = sns.barplot(x="Year", y="MatchesPlayed", data=world_cups, palette="Blues")
g2.set_title("NUMBER OF MATCHS PER CUP", fontsize=14)

plt.subplot(224)
g2 = sns.barplot(x="Year", y="GoalsScored", data=world_cups, palette="Blues")
g2.set_title("NUMBER OF GOALS PER CUP", fontsize=14)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()
winner_by_score_home = world_cups_matches['Home Team Goals'] > world_cups_matches['Away Team Goals']
winner_by_score_away = world_cups_matches['Home Team Goals'] < world_cups_matches['Away Team Goals']
win_by_score = winner_by_score_home | winner_by_score_away
win_penalties = world_cups_matches['Win conditions'].str.len() > 1

win_team_home = np.where(winner_by_score_home, world_cups_matches['Home Team Name'], '')
win_team_away = np.where(winner_by_score_away, world_cups_matches['Away Team Name'], '')

win_cond = world_cups_matches['Win conditions'].str.split(pat='\(|\)|-', expand=True)
win_team_penalties = np.where(win_cond[0].str.len() > 1, 
                     np.where(win_cond[1] > win_cond[2], 
                              world_cups_matches['Home Team Name'], world_cups_matches['Away Team Name']), '')

win_team = np.where(win_team_home != '', win_team_home, 
                    np.where(win_team_away != '', win_team_away, win_team_penalties))

world_cups_matches.loc[:,'result'] = np.where(win_by_score, 'win', np.where(win_penalties, 'win', 'draw'))
world_cups_matches.loc[:,'Winner'] = win_team
world_cups_matches.loc[:,'Looser'] = np.where(world_cups_matches['result'] != 'draw', 
                                        np.where(win_team == world_cups_matches['Home Team Name'], 
                                                 world_cups_matches['Away Team Name'],
                                                 world_cups_matches['Home Team Name']), '')
cup_mask = np.array(Image.open("../input/mask-image-fifa-cup/fifa-cup.jpg"))
#footballer_mask = np.array(Image.open("../input/mask-image-fifa-cup/footballer.jpg"))
#ball_mask = np.array(Image.open("../input/mask-image-fifa-cup/ball.jpg"))

wc_cup = WordCloud(background_color="white", max_words=2000, mask=cup_mask)
#wc_footballer = WordCloud(background_color="white", max_words=2000, mask=footballer_mask)
#wc_ball = WordCloud(background_color="white", max_words=2000, mask=ball_mask)

winner_text = ' '.join(world_cups_matches['Winner'].dropna().tolist())

wc_cup.generate(winner_text)

plt.figure(figsize = (21,12))
sns.set_style("whitegrid")

plt.title('Word cloud of the team that have the most wins', fontsize=14)
plt.imshow(wc_cup, interpolation='bilinear')
plt.axis("off")

plt.show()
home_team_goal = world_cups_matches.groupby(['Year', 'Home Team Name'])['Home Team Goals'].sum()
away_team_goal = world_cups_matches.groupby(['Year', 'Away Team Name'])['Away Team Goals'].sum()
team_goal = pd.concat([home_team_goal, away_team_goal], axis=1)
team_goal = team_goal.fillna(0)
team_goal['goals'] = team_goal['Home Team Goals'] + team_goal['Away Team Goals']
team_goal = team_goal.drop(['Home Team Goals', 'Away Team Goals'], axis=1)
team_goal = pd.DataFrame.from_dict(team_goal.to_dict()).reset_index().rename(index=str, columns={'level_0':'Year', 'level_1':'Team'})

team_goal = team_goal.sort_values(by=['Year', 'goals'], ascending=[True, False])
team_goal_top_5 = team_goal.groupby('Year').head(5)
x, y = team_goal['Year'].values, team_goal['goals'].values

data = []

for team in team_goal_top_5['Team'].drop_duplicates().values :
    year = team_goal_top_5[team_goal_top_5['Team'] == team]['Year']
    goals = team_goal_top_5[team_goal_top_5['Team'] == team]['goals']
    data.append(
        go.Bar(
            x=year,
            y=goals,
            name = team,
        )
    )

layout = go.Layout(
    barmode = "stack", 
    title = "Top 5 teams which scored the most goals",
    showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='pyplot-fifa')