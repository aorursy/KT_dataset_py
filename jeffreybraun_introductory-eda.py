import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
init_notebook_mode()

df = pd.read_csv('/kaggle/input/beat-bobby-flay-results-of-over-300-episodes/beatbobbyflay.csv')
df.winner.value_counts()['Bobby Flay'] / df.shape[0]
judges = list(df.judge1.unique())
judges.extend(list(df.judge2.unique()))
judges.extend(list(df.judge3.unique()))
judges = list(set(judges))

appearances = []
bobby_win = []
bobby_lose = []
bobby_win_percentage = []

for judge in judges:
    results = df[df[['judge1', 'judge2','judge3']].isin([judge]).any(axis=1)].winner.value_counts()
    appearances.append(results.sum())
    if "Bobby Flay" in results:
        bwin = results["Bobby Flay"]
        blose = results.sum() - bwin
    else:
        bwin = 0
        blose = results.sum()
    bobby_win.append(bwin)
    bobby_lose.append(blose)
    bobby_win_percentage.append(bwin/results.sum())
    
df_judge = pd.DataFrame.from_dict({
    'judge':judges,
    'appearances':appearances,
    'bobby_win':bobby_win,
    'bobby_lose':bobby_lose,
    'bobby_win_percentage':bobby_win_percentage
})
    
df_judge = df_judge[df_judge.appearances >= 5]
df_judge.sort_values(by = ['bobby_win_percentage'], inplace=True, ascending=False)
fig = px.bar(df_judge, x = 'judge', y = 'bobby_win_percentage', color = 'appearances', hover_data = ['appearances','bobby_win_percentage'])
fig.update_layout(xaxis_title ='Judge',
                  yaxis_title = 'Bobby Flay Win Percentage',
                 yaxis_tickformat = ',.0%')
fig.show()
guests = list(df.guest1.unique())
guests.extend(list(df.guest2.unique()))
guests.extend(list(df.guest3.unique()))
guests = list(set(guests))
guests = [x for x in guests if str(x) != 'nan']

appearances = []
bobby_win = []
bobby_lose = []
contestant_win_percentage = []

for guest in guests:
    results = df[df[['guest1', 'guest2','guest3']].isin([guest]).any(axis=1)].winner.value_counts()
    appearances.append(results.sum())
    if "Bobby Flay" in results:
        bwin = results["Bobby Flay"]
        blose = results.sum() - bwin
    else:
        bwin = 0
        blose = results.sum()
    bobby_win.append(bwin)
    bobby_lose.append(blose)
    contestant_win_percentage.append(blose/results.sum())
    
df_guest = pd.DataFrame.from_dict({
    'guest_judge':guests,
    'appearances':appearances,
    'bobby_win':bobby_win,
    'bobby_lose':bobby_lose,
    'contestant_win_percentage':contestant_win_percentage
})
    
df_guest = df_guest[df_guest.appearances >= 5]
df_guest.sort_values(by = ['contestant_win_percentage'], inplace=True, ascending=False)
fig = px.bar(df_guest, x = 'guest_judge', y = 'contestant_win_percentage', color = 'appearances', hover_data = ['appearances','contestant_win_percentage'])
fig.update_layout(xaxis_title ='Guest Judge',
                  yaxis_title = 'Contestant Win Percentage',
                 yaxis_tickformat = ',.0%')
fig.show()

def check_bobby_win(winner):
    if winner == 'Bobby Flay':
        return 1
    else:
        return 0

df['bobby_win'] = df['winner'].apply(lambda x: check_bobby_win(x))
df_season = df[['season','bobby_win']].groupby(['season']).agg('mean').reset_index()
fig = px.bar(df_season, x = 'season', y = 'bobby_win', color = 'bobby_win')
fig.update_layout(xaxis_title ='Season',
                  yaxis_title = 'Bobby Flay Win Percentage',
                 yaxis_tickformat = ',.0%')
fig.show()
df_ingredients = pd.read_json('/kaggle/input/recipe-ingredients-dataset/train.json')
def get_cuisine(ingredient):
    cuisine_makeup = dict.fromkeys(list(df_ingredients.cuisine.unique()),0)
    for i in range(df_ingredients.shape[0]):
        ingredients = df_ingredients.loc[i, 'ingredients']
        if ingredient in ingredients:
            cuisine_makeup[df_ingredients.loc[i, 'cuisine']] = cuisine_makeup[df_ingredients.loc[i, 'cuisine']] + 1
    return cuisine_makeup

def create_cuisine_feature(cuisine_makeup, cuisine):
    return cuisine_makeup[cuisine]


df['cuisine'] = df['ingredients'].apply(lambda x: get_cuisine(x))

for cuisine in list(df_ingredients.cuisine.unique()):
    df[cuisine] = df['cuisine'].apply(lambda x: create_cuisine_feature(x, cuisine))        
train_columns = list(df_ingredients.cuisine.unique())

df_train = df[train_columns]
df_train = df_train.div(df_train.sum(axis=1), axis=0)

df_label = df['bobby_win']
from sklearn.linear_model import LogisticRegression

df_label = df_label[df_train.italian.notnull()]
df_train = df_train[df_train.italian.notnull()]

clf = LogisticRegression(random_state=0).fit(df_train, df_label)
print("Logistic Regression score:")
print(clf.score(df_train, df_label))
fig = px.bar(x = list(df_train.columns), y= list(clf.coef_))
fig.show()
