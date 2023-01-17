import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px
import plotly
import re
import os
print("")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print('modules are imported')
df_20 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv', error_bad_lines=False)
df_20.head()
df_20.shape
df_20.columns
useless_columns = ['dob','sofifa_id', 'player_url', 'long_name', 'body_type', 'real_face', 
                   'loaned_from', 'nation_position', 'nation_jersey_number']
df_20  = df_20.drop(useless_columns,axis=1)
df_20.head()
df_20['BMI'] = df_20['weight_kg'] / (df_20['height_cm'] /100)**2
df_20.head()
df_20[['short_name','player_positions']]
new_player_pos = df_20['player_positions'].str.get_dummies(sep =', ').add_prefix('Position_')
new_player_pos.head()
df_20 = pd.concat([df_20,new_player_pos], axis=1)
df_20.head()
df_20 = df_20.drop('player_positions', axis=1)
df_20.head()
columns = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
       'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb',
       'lcb', 'cb', 'rcb', 'rb']
df_20[columns].head()

for col in columns:
    df_20[col] = df_20[col].str.split('+', n=1, expand= True)[0]

df_20[columns].head()
df_20[columns] = df_20[columns].fillna(0)
df_20[columns] = df_20[columns].astype(int)
df_20[columns].head()
columns2 = ["dribbling", "defending", "physic", "passing", "shooting", "pace"]
df_20[columns2]
df_20[columns2].isna().sum()
for c in columns2:
    df_20[c] = df_20[c].fillna(df_20[c].median())
df_20[columns2]
df_20 =df_20.fillna(0)
df_20.isnull().sum()
#using plotly
fig = go.Figure(
    
    data = go.Scatter(
        x= df_20['overall'],
        y= df_20['value_eur'],
        mode= 'markers',
        marker = dict(
            size = 10,
            color = df_20['age'],
            showscale = True
        ),
        text = df_20['short_name'])
)
    
fig.update_layout(title=' Scatter Plot ( colored by  age) year 2020 - Overall ratings vs market value in euro'
                  , xaxis_title = 'Overall Rating'
                  , yaxis_title= 'Market Value in Euros')
fig.show()
fig = px.pie(df_20, names = 'preferred_foot', title= '% of players preferred foot') 
fig.show()
fig = px.histogram(df_20, x= 'age' , title= 'Histogram of Players Ages')
fig.show()
df_15 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_15.csv", error_bad_lines=False)
df_16 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_16.csv", error_bad_lines=False)
df_17 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_17.csv", error_bad_lines=False)
df_18 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_18.csv", error_bad_lines=False)
df_19 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_19.csv", error_bad_lines=False)
attributes = ['Pace','Shooting','Passing','Dribbling','Defending','Physic','Overall'] 
def playergrowth(name):
    
    data20 = df_20[df_20.short_name.str.startswith(name)]
    data19 = df_19[df_19.short_name.str.startswith(name)]
    data18 = df_18[df_18.short_name.str.startswith(name)]
    data17 = df_17[df_17.short_name.str.startswith(name)]
    data16 = df_16[df_16.short_name.str.startswith(name)]
    data15 = df_15[df_15.short_name.str.startswith(name)]
    
    
    trace0 = go.Scatterpolar(
        
        r = [ data20['pace'].values[0], data20['shooting'].values[0], data20['passing'].values[0]
           , data20['dribbling'].values[0],  data20['defending'].values[0], data20['physic'].values[0]
           , data20['overall'].values[0]]
        ,
        theta = attributes,
        fill = 'toself',
        name = '2020'
    
    )
    trace1 = go.Scatterpolar(
        
        r=[data19['pace'].values[0], data19['shooting'].values[0],
           data19['passing'].values[0] , data19['dribbling'].values[0],  
           data19['defending'].values[0], 
           data19['physic'].values[0]  , data19['overall'].values[0] ],
        theta = attributes,
        fill = 'toself',
        name = '2019'
    
    )
    
    trace2 = go.Scatterpolar(
        
        r = [data18['pace'].values[0], data18['shooting'].values[0], 
             data18['passing'].values[0], data18['dribbling'].values[0], 
             data18['defending'].values[0], data18['physic'].values[0]
           , data18['overall'].values[0] ],
        theta = attributes,
        fill = 'toself',
        name = '2018'
    
    )
    trace3 = go.Scatterpolar(
        
        r  = [ data17['pace'].values[0], data17['shooting'].values[0], data17['passing'].values[0]
           , data17['dribbling'].values[0],  data17['defending'].values[0], data17['physic'].values[0]
           , data17['overall'].values[0] ],
        theta = attributes,
        fill = 'toself',
        name = '2017'
    
    )
    trace4 = go.Scatterpolar(
        
        r = [data16['pace'].values[0], data16['shooting'].values[0], data16['passing'].values[0]
           , data16['dribbling'].values[0],  data16['defending'].values[0], data16['physic'].values[0]
           , data16['overall'].values[0] ],
        theta = attributes,
        fill = 'toself',
        name = '2016'
    
    )
    
    trace5 = go.Scatterpolar(
        
        r = [data15['pace'].values[0], data15['shooting'].values[0], data15['passing'].values[0]
           , data15['dribbling'].values[0],  data15['defending'].values[0], data15['physic'].values[0]
           , data15['overall'].values[0] ],
        theta = attributes,
        fill = 'toself',
        name = '2015'
    
    )
    
    data = [ trace0, trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(
            polar = dict(
            radialaxis = dict(
            visible = True,
            range = [0,100]))
            , 
            showlegend = True, 
            title = 'Stats retaled to {} from 2015 to 2020'.format(name) )
    
    fig = go.Figure(data = data , layout = layout)
    fig.show()
    
    
playergrowth('L. Messi')
playergrowth('L. Suárez')
playergrowth('Neymar')
playergrowth('Cristiano Ronaldo')
playergrowth('K. Benzema')
playergrowth('G. Bale')
attack = ['RW', 'LW', 'ST', 'CF', 'LS', 'RS', 'RF', 'LF']

sample_att = df_20.query('team_position in @attack')
sample_att.head()
fig = px.pie(sample_att, names='team_position', color_discrete_sequence= px.colors.sequential.Magma_r, 
            title = '% of Players in Attacker Position')
fig.show()
mid = ['CAM', 'RCM', 'CDM', 'LDM', 'RM', 'LCM', 'LM', 'RDM', 'RAM','CM', 'LAM']

sample_mid = df_20.query('team_position in @mid')
sample_mid.head()
fig = px.pie(sample_mid, names='team_position', color_discrete_sequence= px.colors.sequential.Blugrn_r ,
            title = '% of Players in MidField Position')
fig.show()
defence = ['LCB', 'RCB', 'LB', 'RB', 'CB', 'RWB', 'LWB']
sample_def = df_20.query('team_position in @defence')
sample_def.head()
fig = px.pie(sample_def, names='team_position', color_discrete_sequence= px.colors.sequential.GnBu_r, 
            title = '% of Players in Defender Position')
fig.show()
def pick_top_player(pos, value):
    column = str('Position_')+str.upper(pos)
    target_player= df_20[(df_20[column] == 1) & (df_20['value_eur'] <= value)][['short_name','age','overall','value_eur']].head(5)
    
    return target_player

                        
pick_top_player('lb',400000000)
pick_top_player('cf',500000000)
pick_top_player('ST',55000000)
