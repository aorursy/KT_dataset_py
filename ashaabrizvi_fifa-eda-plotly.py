import pandas as pd

import numpy as np

import plotly

import plotly.graph_objects as go

import plotly.express as px
fifa_20 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
fifa_20.head()
fifa_20.shape # To check no of rows and columns
col = list(fifa_20.columns)  # To print all the columns

print(col)
useless_column = ['dob','sofifa_id','player_url','long_name','body_type','real_face','nation_position','loaned_from','nation_jersey_number']
fifa_20 = fifa_20.drop(useless_column, axis = 1)
fifa_20.shape # To check how many columns did we dropped
fifa_20['BMI'] = fifa_20 ['weight_kg'] / (fifa_20['height_cm'] / 100) ** 2
fifa_20.head()
fifa_20[['short_name','player_positions']]
new_player_position = fifa_20['player_positions'].str.get_dummies(sep=',').add_prefix('Position')

new_player_position.head()
fifa_20 =  pd.concat([fifa_20,new_player_position],axis = 1)
fifa_20 =  fifa_20.drop('player_positions',axis=1)
columns = ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm', 'cdm','rdm','rwb','lb','lcb','cb','rcb','rb']
fifa_20[columns].head()
for col in columns:

  fifa_20[col]=fifa_20[col].str.split('+',n=1,expand = True)[0]



fifa_20[columns]
fifa_20[columns] = fifa_20[columns].fillna(0)
fifa_20[columns] = fifa_20[columns].astype(int)
columns = ['dribbling','defending','physic','passing','shooting','pace']
fifa_20[columns]
fifa_20[columns].isnull().sum()
for col in columns:

  fifa_20[col] = fifa_20[col].fillna(fifa_20[col].median())

fifa_20[columns]
fifa_20 = fifa_20.fillna(0)
fifa_20.isnull().sum() # To check null values
# Scatter Plot in Plotly

fig = go.Figure(data=go.Scatter(x=fifa_20['overall'],y=fifa_20['value_eur'],mode='markers',marker=dict(size=10,color=fifa_20['age'],showscale=True),text=fifa_20['short_name']))

fig.update_layout(title='Scatter Plot for Overall Rating v Value in Euros',xaxis_title='Overall Rating',yaxis_title='Value in Euros')



fig.show()
# Scatter Plot

fig = go.Figure(data=go.Scatter(x=fifa_20['overall'],y=fifa_20['BMI'],mode='markers',marker=dict(size=10,color=fifa_20['age'],showscale=True),text=fifa_20['short_name']))

fig.update_layout(title='Scatter Plot for Overall Rating v BMI of a Player',xaxis_title='Overall Rating',yaxis_title='BMI of a Player')

fig.show()
# Scatter Plot

fig = go.Figure(data=go.Scatter(x=fifa_20['potential'],y=fifa_20['wage_eur'],mode='markers',marker=dict(size=10,color=fifa_20['age'],showscale=True),text=fifa_20['short_name']))

fig.update_layout(title='Scatter Plot for Potential Rating v Wage in Euros',xaxis_title='Potential Rating',yaxis_title='Wage in Euros')

fig.show()
# Analysis of Preferred Foot through a Pie Chart

fig = px.pie(fifa_20,names='preferred_foot',title='Percentage of Players as per there preferred Foot')

fig.show()
# Pie Chart

fig = px.pie(fifa_20.head(25),names='club',title='Percentage of Clubs in Top 25 FIFA Players')

fig.show()
# Pie Chart

fig = px.pie(fifa_20.head(25),names='nationality',title='Percentage of Nations in Top 25 FIFA Players')

fig.show()
# Bar Charts

fig = px.bar(fifa_20.head(10), y='potential',x='short_name',color='age',

             labels={'Overall Rating v Nation of Top 20'}, height=400)

fig.update_layout(title='Comparison of Potential of Top 10 FIFA Players',xaxis_title='Player Name',yaxis_title='Potential')

fig.show()
# 3D Plots

fig = px.scatter_3d(fifa_20.head(20), x='potential', y='overall', z='wage_eur',

              color='short_name')

fig.update_layout(title='3D Plot of Potential, Overall and Wage in Euros of Top 20 FIFA Players')

fig.show()
# 3D Plots

fig = px.scatter_3d(fifa_20.head(20), x='potential', y='overall', z='value_eur',

              color='short_name')

fig.update_layout(title='3D Plot of Potential, Overall and Value in Euros of Top 20 FIFA Players')

fig.show()