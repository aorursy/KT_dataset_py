# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Manipulation 

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import seaborn as sns



#datetime

import datetime as dt



#warnings

import warnings

warnings.filterwarnings("ignore")





#plotly

import plotly.graph_objects as go

import plotly.figure_factory as ff

import plotly.express as px

from plotly.subplots import make_subplots
df20 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

df19 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_19.csv')

df18 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_18.csv')

df17 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_17.csv')

df16 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_16.csv')

df15 = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_15.csv')

df20['count'] = 1  
def top_n_charts(field, n):

    df_club = df20.groupby([field]).mean()

    df_club = df_club.sort_values('overall', ascending = False).reset_index()



    ls = df20.groupby([field]).sum()

    ls = ls[ls['count'] > 10].index



    df_club = df_club[df_club[field].isin(ls)]



    f, ax = plt.subplots(figsize = (20,5))

    sns.barplot(x = field, y = 'overall', data = df_club.iloc[:n])

    ax.set(ylim = (60,85))



top_n_charts('club', 10)

top_n_charts('nationality', 10)
df20_pot = df20[(df20.age.astype(int) >= 18) & (df20.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df20_over = df20[(df20.age.astype(int) >= 18) & (df20.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df20_summary = pd.concat([df20_pot, df20_over], axis=1)

df19_pot = df19[(df19.age.astype(int) >= 18) & (df19.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df19_over = df19[(df19.age.astype(int) >= 18) & (df19.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df19_summary = pd.concat([df19_pot, df19_over], axis=1)

df18_pot = df18[(df18.age.astype(int) >= 18) & (df18.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df18_over = df18[(df18.age.astype(int) >= 18) & (df18.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df18_summary = pd.concat([df18_pot, df18_over], axis=1)

df17_pot = df17[(df17.age.astype(int) >= 18) & (df17.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df17_over = df17[(df17.age.astype(int) >= 18) & (df17.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df17_summary = pd.concat([df17_pot, df17_over], axis=1)

df16_pot = df16[(df16.age.astype(int) >= 18) & (df16.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df16_over = df16[(df16.age.astype(int) >= 18) & (df16.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df16_summary = pd.concat([df16_pot, df16_over], axis=1)

df15_pot = df15[(df15.age.astype(int) >= 18) & (df15.age.astype(int) <= 35)].groupby(['age'])['potential'].mean()

df15_over = df15[(df15.age.astype(int) >= 18) & (df15.age.astype(int) <= 35)].groupby(['age'])['overall'].mean()

df15_summary = pd.concat([df15_pot, df15_over], axis=1)



fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, figsize=(10, 25))

ax1.plot(df20_summary)

ax1.set_ylabel('Rating')

ax1.set_title('FIFA 20 - Average Rating by Age')

ax2.plot(df19_summary)

ax2.set_ylabel('Rating')

ax2.set_title('FIFA 19 - Average Rating by Age')

ax3.plot(df18_summary)

ax3.set_ylabel('Rating')

ax3.set_title('FIFA 18 - Average Rating by Age')

ax4.plot(df17_summary)

ax4.set_ylabel('Rating')

ax4.set_title('FIFA 17 - Average Rating by Age')

ax5.plot(df16_summary)

ax5.set_ylabel('Rating')

ax5.set_title('FIFA 16 - Average Rating by Age')

ax6.plot(df15_summary)

ax6.set_ylabel('Rating')

ax6.set_title('FIFA 15 - Average Rating by Age')
df20['best_position'] = df20['player_positions'].str.split(',').str[0]



def get_best_squad(df_name, position):

    df_copy = df_name.copy()

    store = []

    for i in position:

        store.append([i,df_copy.loc[[df_copy[df_copy['best_position'] == i]['overall'].idxmax()]]['short_name'].to_string(index = False), df_copy[df_copy['best_position'] == i]['overall'].max()])

        df_copy.drop(df_copy[df_copy['best_position'] == i]['overall'].idxmax(), inplace = True)

    #return store

    return pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)



squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

print ('4-3-3 in FIFA 20')

print (get_best_squad(df20, squad_433))
def get_top20_players(df_name):

    df_copy = df_name.sort_values(['overall', 'potential'], ascending=[False, False]).head(20)

    store = []

    for index, row in df_copy.iterrows():

        store.append([row['best_position'], row['short_name'], row['overall'], row['potential'], row['age']])

    return np.mean([x[2] for x in store]).round(1), np.mean([x[3] for x in store]).round(1), np.mean([x[4] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(20, 5), columns = ['Position', 'Player', 'Overall', 'Potential', 'Age']).to_string(index = False)



top20_players20_overall, top20_players20_potential, top20_players20_age, top20_players20 = get_top20_players(df20)

print('FIFA 20 - Top 20 Players')

print('Average overall: {:.1f}'.format(top20_players20_overall))

print('Average potential: {:.1f}'.format(top20_players20_potential))

print('Average age: {:.1f}'.format(top20_players20_age))

print(top20_players20)
fifa_potential=df20[(df20.potential>85 )& (df20.overall>80)]
fifa_potential_ready=fifa_potential[(fifa_potential.overall<100)&(fifa_potential.overall>70)]
position="ST"

fifa_potential_st=fifa_potential_ready[df20.player_positions.str.contains(position)]
fifa_potential_st
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_st.short_name, y=fifa_potential_st.overall,text=fifa_potential_st.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_st.short_name, y=fifa_potential_st.potential,text=fifa_potential_st.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential ST in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_st.short_name, y=fifa_potential_st.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential ST player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()