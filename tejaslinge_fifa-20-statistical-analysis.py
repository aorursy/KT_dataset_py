import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 150)

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fifa_original = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

fifa_original.head()
fifa_original.set_index('short_name', inplace = True)
position_attributes = fifa_original.loc[:, 'ls':].dropna()

position_attributes.columns  = [i.upper() for i in position_attributes.columns]

position_attributes.head()
gk_attributes = fifa_original[['overall', 'potential', 'gk_diving', 'gk_handling', 'gk_speed', 'gk_reflexes', 'gk_kicking', 'gk_positioning', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']]

gk_attributes = gk_attributes.dropna()



gk_attributes.head()
gk_attributes.corr()
plt.figure(figsize = (28,12))

sns.set_context('poster',font_scale=1)

sns.heatmap(gk_attributes.corr(), annot = True).set_title('GK Attributes')
defense_pos = np.array(['CB','LB','RB'])



defenders = fifa_original[fifa_original['player_positions'].isin(i for i in defense_pos)]



defense_attributes = defenders[['overall', 'potential', 'defending', 'physic', 'pace', 'passing', 'dribbling', 'attacking_heading_accuracy', 'attacking_short_passing', 'skill_long_passing', 'skill_ball_control', 'movement_sprint_speed', 'movement_acceleration', 'movement_reactions', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle',]]

defense_attributes= defense_attributes.dropna()

defense_attributes.corr()  
plt.figure(figsize = (28,12))

sns.set_context('poster',font_scale=.75)

sns.heatmap(defense_attributes.corr(), annot = True, vmin = -1, vmax = 1, cmap= 'coolwarm')

attack_pos = np.array(['LS','RS','ST','CF','CAM','RM','LM','RW','LW','LAM','RAM'])

attackers = fifa_original[fifa_original['player_positions'].isin(i for i in attack_pos)]

attack = attackers[['overall', 'potential', 'skill_moves', 'pace','shooting', 'passing', 'dribbling', 'physic',

                   'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',

                   'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',

                   'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility',

                   'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina',

                   'power_strength', 'power_long_shots']]

attack = attack.dropna()
attack_corr = attack.corr()

attack_corr
fig = go.Figure(data=go.Heatmap(

                   z=attack_corr,

                   x=attack_corr.columns,

                   y=attack_corr.columns, colorscale='Blues',

                   hoverongaps = True))

fig.show()
fifa_original = fifa_original[['long_name', 'age', 'club', 'nationality', 'overall', 'potential', 'value_eur',

             'wage_eur', 'player_positions', 'preferred_foot', 'international_reputation', 'weak_foot', 'skill_moves', 'work_rate', 'body_type',

             'release_clause_eur', 'team_position', 'team_jersey_number', 'nation_position', 'nation_jersey_number', 'pace', 'shooting', 'passing', 'dribbling', 'defending',

             'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',

             'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',

             'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',

             'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties',

             'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_handling']]



fifa_original.columns = ['Full Name', 'Age', 'Club', 'Nationality', 'Overall', 'Potential', 'Value(Euro)', 'Wage(Euro)', 'Position(s)',

                'Foot', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate', 'Body Type', 'Release Clause', 'Team Pos', 'Jersey No.', 'National Pos', 'National Jersey No.',

                'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic'] + [str(i) for i in fifa_original.columns[26:]] 

fifa_original.head()
fifa_original['Club'].fillna('No Club', inplace = True)

fifa_original['Position(s)'].fillna('unknown', inplace = True)

fifa_original['Work Rate'].fillna('Medium/ Medium', inplace = True)

fifa_original['Foot'].fillna('Right', inplace = True)



fifa_original.head()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = fifa_original['Overall'],

    y = fifa_original['Value(Euro)'],

    mode='markers',

    marker=dict(

        size=16,

        color=fifa_original['Age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= fifa_original.index,

))



fig.update_layout(title='Styled Scatter Plot (colored by Age) year 2020 - Overall Rating vs Value in Euros',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(family='Cambria, monospace', size=12, color='#000000'))

fig.show()
TopClubsInVal = fifa_original[['Club', 'Value(Euro)']]

TopClubsInVal = TopClubsInVal.set_index(n for n in range(TopClubsInVal.shape[0]))

TopClubsInVal = pd.DataFrame(TopClubsInVal.groupby('Club')['Value(Euro)'].sum()).sort_values('Value(Euro)', ascending = False).head(20)



fig = go.Figure(

        data = [go.Bar(y = TopClubsInVal['Value(Euro)'],

                       x = TopClubsInVal.index)],

        layout_title_text = "Top 20 Clubs by Total Player Value(Euro) of Fifa 20"

        

)

fig.update_traces(marker_color='green')

fig.show() 
most_valued = fifa_original[['Value(Euro)', 'Club', 'Nationality']]

most_valued = most_valued.sort_values('Value(Euro)', ascending = False).head(30)



fig = go.Figure(

        data = [go.Bar(y = most_valued['Value(Euro)'],

                       x = most_valued.index)],

                       

        

        layout_title_text = 'Highest Valued Players of Fifa 20'

)



fig.show()
top_ratings = fifa_original[['Club', 'Overall', 'Potential', 'Value(Euro)']]

top_ratings = top_ratings.sort_values('Overall', ascending = False).head(30)



fig = go.Figure(

        data = [go.Bar(y = top_ratings['Overall'],

                       x = top_ratings.index)],

        layout_title_text = 'Top Rated Players of Fifa 20'

)

fig.update_traces(marker_color='#00A')

fig.show()
GK = fifa_original[fifa_original['Position(s)'] == 'GK']

GK = GK[['Club', 'Nationality', 'Overall', 'Potential']]
GK = GK.sort_values('Overall', ascending = False).head(20)



fig = go.Figure(

        data = [go.Bar(y = GK['Overall'],

                       x = GK.index)],

        layout_title_text = "Top Rated GK's of Fifa 20"

)

fig.update_traces(marker_color='goldenrod')

fig.show()
sns.set(style = 'darkgrid')

plt.figure(figsize = (35,15))

sns.set_context('poster',font_scale=1.5)

sns.countplot(x = 'Foot', data = fifa_original, palette = 'coolwarm').set_title('Left v/s Right')
best_fk = fifa_original[['skill_fk_accuracy']].sort_values('skill_fk_accuracy', ascending = False).head(30)



x = best_fk.skill_fk_accuracy

plt.figure(figsize=(12,8))

plt.style.use('seaborn-paper')

sns.set_context('poster',font_scale=.5)



ax = sns.barplot(x = x, y = best_fk.index, data = best_fk)

ax.set_xlabel(xlabel = "Free-Kick Attributes", fontsize = 16)

ax.set_ylabel(ylabel = 'Player Name(s)', fontsize = 16)

ax.set_title(label = "Bar Plot of Best Free-Kick Takers", fontsize = 20)

plt.show()



top_clubs = np.array(['Real Madrid', 'Manchester City', 'Tottenham Hotspur', 'Napoli',

             'FC Barcelona', 'Juventus', 'Paris Saint-Germain', 'Liverpool',

             'Manchester United', 'Chelsea', 'Atlético Madrid', 'Arsenal',

             'Borussia Dortmund', 'FC Bayern MÃ¼nchen', 'West Ham United', 'FC Schalke 04',

             'Roma', 'Leicester City', 'Inter', 'Milan'])
top_clubs.shape
condition_top20 = ((fifa_original['Potential'] - fifa_original['Overall']) >= 15) & (fifa_original['Potential'] >= 80) & (fifa_original['Club'].isin(top_clubs))  & (fifa_original['Age'] <= 20) 

young_players_top20 = fifa_original[condition_top20].sort_values('Potential', ascending = False)



young_clubs_top20 = young_players_top20.Club

young_clubs_top20 = pd.DataFrame(young_clubs_top20.value_counts())

young_players_top20.shape
condition_all = ((fifa_original['Potential'] - fifa_original['Overall']) >= 15) & (fifa_original['Potential'] >= 80)   & (fifa_original['Age'] <= 20) 

young_players_all = fifa_original[condition_all].sort_values('Potential', ascending = False)



young_clubs_all = young_players_all.Club

young_clubs_all = pd.DataFrame(young_clubs_all.value_counts())

young_players_all.shape
young_players_top20.head()
young_players_all.head()
fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],

                    subplot_titles=['From Top 20 Clubs', 'From All Clubs'])

fig.add_trace(go.Pie(labels=young_clubs_top20.index, values= young_clubs_top20.Club, scalegroup='one',

                     name="From Top 20 Clubs"), 1, 1)

fig.add_trace(go.Pie(labels=young_clubs_all.head(20).index, values=young_clubs_all.head(20).Club, scalegroup='one',

                     name="From All Clubs"), 1, 2)



fig.update_layout(title_text='Count of Promising Young Players')

fig.show()
young_players_all_cheapest = young_players_all[['Age', 'Club', 'Nationality', 'Overall', 'Potential', 'Value(Euro)', 'Wage(Euro)', 'Position(s)']].sort_values('Value(Euro)', ascending = True) 



fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = young_players_all_cheapest['Potential'],

    y = young_players_all_cheapest['Value(Euro)'],

    mode='markers',

    marker=dict(

        size=16,

        color=young_players_all_cheapest['Age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= young_players_all_cheapest.index,

))



fig.update_layout(title='Scatter Plot of Potential of Promising Young Players vs Value in Euros',

                  xaxis_title='Potential',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(family='Cambria, monospace', size=12, color='#000000'))

fig.show()
young_players_all_cheapest50 = young_players_all_cheapest.head(50)



fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = young_players_all_cheapest50['Potential'],

    y = young_players_all_cheapest50['Value(Euro)'],

    mode='markers',

    marker=dict(

        size=16,

        color=young_players_all_cheapest50['Age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= young_players_all_cheapest50.index,

))



fig.update_layout(title='Scatter Plot of Potential of 50 Cheapest Promising Young Players vs Value in Euros',

                  xaxis_title='Potential',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(family='Cambria, monospace', size=12, color='#000000'))

fig.show()