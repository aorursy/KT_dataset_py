import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%config InlineBackend.figure_format = 'retina'

pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv')

df.head()
df.info()
# 'date' to datetime

df['date'] = pd.to_datetime(df['date'])

df['Match_year'] = df['date'].dt.year

df['Match_month'] =df['date'].dt.month

df.head()
# Define Match result function



def Match_result(df):

    

    # Win & Lose & Draw

    home_win = df[(df['home_score'] - df['away_score']) > 0]['home_team'].value_counts()

    home_lose = df[(df['home_score'] - df['away_score']) < 0]['home_team'].value_counts()

    away_win = df[(df['home_score'] - df['away_score']) < 0]['away_team'].value_counts()

    away_lose = df[(df['home_score'] - df['away_score']) > 0]['away_team'].value_counts()

    home_draw = df[(df['home_score'] - df['away_score']) == 0]['home_team'].value_counts()

    away_draw = df[(df['home_score'] - df['away_score']) == 0]['away_team'].value_counts()

    

    # Total match

    # Scoring

    # GF : Goal For

    # GA : Goal Against

    

    # Total GF = Home_GF + Away_GF

    Home_GF = df.groupby('home_team')['home_score'].sum()

    Away_GF = df.groupby('away_team')['away_score'].sum()

    

    # Total GA = Home_GA + Away_GA

    Home_GA = df.groupby('home_team')['away_score'].sum() 

    Away_GA = df.groupby('away_team')['home_score'].sum()

    

    result_data = pd.DataFrame({'W' : home_win.add(away_win, fill_value = 0),

                            'L' : home_lose.add(away_lose, fill_value = 0),

                           'D' : home_draw.add(away_draw, fill_value = 0),

                            'Home W' : home_win,

                                'Home L' : home_lose,

                                'Away W' : away_win,

                                'Away L' : away_lose,

                                'Home D' : home_draw,

                                'Away D' : away_draw,

                                'Home GF' : Home_GF,

                                'Away GF' : Away_GF,

                                'Home GA' : Home_GA,

                                'Away GA' : Away_GA,

                                'Total GF' : Home_GF.add(Away_GF, fill_value = 0),

                                'Total GA' : Home_GA.add(Away_GA, fill_value = 0)

                               })

    

    result_data.fillna(0, inplace= True)

    result_data = result_data.reset_index().rename(columns = {'index' : 'country'} )

    

    # Add Total match

    result_data['Total Played'] = result_data['W'] + result_data['L'] + result_data['D']

    

    return result_data
Match_result(df).head()
# Define Top10 of wins



def Top10_win(df):

    df = df.sort_values('W', ascending= False)

    

    fig = go.Figure()

    fig.add_trace(go.Bar(x = df['country'].head(10), y = df['W'].head(10), name = 'W'))

    fig.add_trace(go.Bar(x = df['country'].head(10), y = df['L'].head(10), name = 'L'))

    fig.add_trace(go.Bar(x = df['country'].head(10), y = df['D'].head(10), name = 'D'))

    

    fig.update_layout(xaxis_title = 'Country', yaxis_title = 'Match result')

    

    

    return fig.show()
# Define Top10 of Percentages of Victories



def Top10_Percent_vic(df):

    df['Percentages of Victories'] = df['W'] / (df['W'] + df['D'] + df['L']) * 100

    df = df.sort_values('Percentages of Victories', ascending= False)

    

    fig = go.Figure()

    fig.add_trace(go.Bar(y = df['country'].head(10), 

                         x = df['Percentages of Victories'].head(10),

                         orientation='h'))

    

    fig.update_layout(yaxis_title = 'Country', xaxis_title = 'Percentages of Victories')

    fig.update_layout(xaxis=dict(range=[df['Percentages of Victories'].head(10).min()-2, 

                                        df['Percentages of Victories'].head(10).max() +5]))

    fig.update_traces(texttemplate='<b>%{x:.1f}', textposition='outside')

    return fig.show()    
plt.figure(figsize = (20,4))

g = sns.countplot(df['Match_year'])

g = plt.xticks(rotation = 90, fontsize = 15)

g = plt.yticks(fontsize = 15)

g = plt.title('Match', fontsize = 15)
plt.figure(figsize = (20,4))

g = sns.countplot(df['Match_month'])

g = plt.xticks(fontsize = 15)

g = plt.yticks(fontsize = 15)

g = plt.title('Match', fontsize = 15)
#how many tournament is there?

plt.figure(figsize = (20,5))

g = sns.countplot(df['tournament'])

g = plt.xticks(rotation  = 90, fontsize = 15)

g = plt.yticks(fontsize = 15)
values = df['tournament'].value_counts()

names= df['tournament'].value_counts().index



fig = px.pie(values = values, names = names, title = 'Tournament Type')

fig.update_traces(textinfo = 'label + percent', marker=dict(line=dict(color='#000000', width=1)))

fig.show()
# UEFA Euro qualification & UEFA Euro

df_UEFA = df.query("(tournament == 'UEFA Euro qualification') | (tournament == 'UEFA Euro')")

df_UEFA.head()
plt.figure(figsize = (20,5))

g = sns.countplot(data = df_UEFA, x= 'Match_year', hue = 'tournament')

g = plt.xticks(rotation  = 90, fontsize = 15)

g = plt.yticks(fontsize = 15)
plt.figure(figsize = (20,5))

g = sns.countplot(data = df_UEFA, x= 'Match_month', hue = 'tournament')

g = plt.xticks(rotation  = 90, fontsize = 15)

g = plt.yticks(fontsize = 15)
# Top 10 of win in UEFA Euro(Incl. UEFA Euro qualification)

UEFA_result = Match_result(df_UEFA)

Top10_win(UEFA_result)
# Top 10 of Percentages of Victories in UEFA Euro(Incl. UEFA Euro qualification)

Top10_Percent_vic(UEFA_result)
# FIFA World Cup

df_FIFA = df.query('tournament == "FIFA World Cup"')

FIFA_result = Match_result(df_FIFA)

Top10_win(FIFA_result)
Top10_Percent_vic(FIFA_result)
# Who is the best team of all time

allresult = Match_result(df)

Top10_win(allresult)

Top10_Percent_vic(allresult)
# Where was country matche was held?

fig = px.scatter_geo(data_frame=df, locations= 'country', 

               locationmode='country names', animation_frame='tournament',

                    title = '<b>Country where match played')

fig.show()
# there was home advantage?

df_HA = Match_result(df.query("neutral == False"))

df_HA.head()
# Total matches percentage of victories(%)

df_HA['%'] = df_HA['W'] / (df_HA['Total Played']) * 100



# Home matches percentage of victories(%)

df_HA['Home %'] = df_HA['Home W'] / (df_HA['Home W'] + df_HA['Home L'] + df_HA['Home D']) * 100

df_HA.fillna(0, inplace= True)
# Average of GF, GA

df_HA['avg GF'] = df_HA['Total GF'] / (df_HA['W'] + df_HA['L'] + df_HA['D'])

df_HA['avg GA'] = df_HA['Total GA'] / (df_HA['W'] + df_HA['L'] + df_HA['D'])

df_HA['home avg GF'] = df_HA['Home GF'] / (df_HA['Home W'] + df_HA['Home L'] + df_HA['Home D'])

df_HA['home avg GA'] = df_HA['Home GA'] / (df_HA['Home W'] + df_HA['Home L'] + df_HA['Home D'])

df_HA.fillna(0, inplace= True)
# Compare Total % vs Home %

avg = df_HA['%'].mean()

Havg = df_HA['Home %'].mean()

g = sns.barplot(x = ['Total matches', 'Home matches'], y = [avg, Havg], palette = 'Blues_d' )

g = plt.title('Total % vs Home %')
avg = df_HA[df_HA['Total Played'] > 50]['%'].mean()

Havg = df_HA[df_HA['Total Played'] > 50]['Home %'].mean()



g = sns.barplot(x = ['Total matches', 'Home matches'], y = [avg, Havg], palette = 'Blues_d' )

g = plt.title('Total % vs Home % over 50 matches')
avg = df_HA[df_HA['Total Played'] < 50]['%'].mean()

Havg = df_HA[df_HA['Total Played'] < 50]['Home %'].mean()



g = sns.barplot(x = ['Total matches', 'Home matches'], y = [avg, Havg], palette = 'Blues_d' )

g = plt.title('Total % vs Home % under 50 matches')
# GF differences(Total GF vs Home GF)



hist_data = [df_HA['avg GF'], df_HA['home avg GF']]



fig = ff.create_distplot(hist_data= hist_data, 

                         group_labels=['Avg GF', 'Home avg GF'], 

                         show_hist = False)

fig.update_layout(title ='<b>Total GF vs Home GF')

fig.show()
# Which countries host the most matches where they themselves are not participating in



fig = px.choropleth(locations= df.query("neutral == True")['country'].value_counts().index,

              locationmode='country names',

              color= df.query("neutral == True")['country'].value_counts(),

                   color_continuous_scale = 'Reds',

                    title = '<b>Countries host not participating',

                   labels = {'color' : 'Match played'})

fig.show()