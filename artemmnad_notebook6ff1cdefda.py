import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3

import numpy as np

from numpy import random



#Загрузка данных из таблиц

with sqlite3.connect('../input/database.sqlite') as con:

    countries = pd.read_sql_query("SELECT * from Country", con)

    matches = pd.read_sql_query("SELECT * from Match", con)

    leagues = pd.read_sql_query("SELECT * from League", con)

    teams = pd.read_sql_query("SELECT * from Team", con)

    

#Выбираем Англию



selected_countries = ['England']

countries = countries[countries.name.isin(selected_countries)]

leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))





#Выбираем необходимые поля



matches = matches[matches.league_id.isin(leagues.id)]

matches = matches[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','B365H', 'B365D' ,'B365A']]

matches.dropna(inplace=True)

matches.head()
from scipy.stats import entropy



def match_entropy(row):

    odds = [row['B365H'],row['B365D'],row['B365A']]

    

    #Меняем шанс на веротность

    probs = [1/o for o in odds]

    

    #Нормализуем к 1

    norm = sum(probs)

    probs = [p/norm for p in probs]

    return entropy(probs)



#Посчитать энтропию для матча

matches['entropy'] = matches.apply(match_entropy,axis=1)



#Посчитать энтропию для домашних команд

entropy_means = matches.groupby(('season','home_team_api_id')).entropy.mean()

entropy_means = entropy_means.reset_index().pivot(index='season', columns='home_team_api_id', values='entropy')

entropy_means.columns = [teams[teams.team_api_id==x].team_long_name.values[0] for x in entropy_means.columns]

entropy_means = entropy_means[['Manchester United', 'Tottenham Hotspur', 'Chelsea']]

entropy_means.head(6)
#График

ax = entropy_means.plot(figsize=(12,8),marker='o')



#Название

plt.title('Home Teams Predictability', fontsize=16)



plt.xticks(rotation=50)



#Установить цвета

colors = [x.get_color() for x in ax.get_lines()]

colors_mapping = dict(zip(leagues.id,colors))



#Удалить название оси Х

ax.set_xlabel('')



#Добавить условное обозначение

plt.legend(loc='lower left')



#Добавить название

ax.annotate('', xytext=(7.2, 0.93),xy=(7.2, 0.969),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('', xytext=(7.2, 0.76),xy=(7.2, 0.721),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('less predictable', xy=(7.3, 0.958), annotation_clip=False,fontsize=14,rotation='vertical')

ax.annotate('more predictable', xy=(7.3, 0.792), annotation_clip=False,fontsize=14,rotation='vertical')
#Посчитать энтропию для гостевых команд

entropy_means = matches.groupby(('season','away_team_api_id')).entropy.mean()

entropy_means = entropy_means.reset_index().pivot(index='season', columns='away_team_api_id', values='entropy')

entropy_means.columns = [teams[teams.team_api_id==x].team_long_name.values[0] for x in entropy_means.columns]

entropy_means = entropy_means[['Manchester United', 'Tottenham Hotspur', 'Chelsea']]

entropy_means.head(6)
#График

ax = entropy_means.plot(figsize=(12,8),marker='o')



#Название

plt.title('Away Teams Predictability', fontsize=16)



plt.xticks(rotation=50)



#Установить цвета

colors = [x.get_color() for x in ax.get_lines()]

colors_mapping = dict(zip(leagues.id,colors))



#Удалить название оси Х

ax.set_xlabel('')



#Добавить условное обозначение

plt.legend(loc='lower left')



#Добавить название

ax.annotate('', xytext=(7.2, 0.93),xy=(7.2, 0.969),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('', xytext=(7.2, 0.76),xy=(7.2, 0.721),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('less predictable', xy=(7.3, 0.958), annotation_clip=False,fontsize=14,rotation='vertical')

ax.annotate('more predictable', xy=(7.3, 0.792), annotation_clip=False,fontsize=14,rotation='vertical')
#График

ax = entropy_means.plot(figsize=(12,8),marker='o')



#Название

plt.title('Home Teams Predictability', fontsize=16)



plt.xticks(rotation=50)



#Установить цвета

colors = [x.get_color() for x in ax.get_lines()]

colors_mapping = dict(zip(leagues.id,colors))



#Удалить название оси Х

ax.set_xlabel('')



#Добавить условное обозначение

plt.legend(loc='lower left')



#Добавить название

ax.annotate('', xytext=(7.2, 0.93),xy=(7.2, 0.969),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('', xytext=(7.2, 0.76),xy=(7.2, 0.721),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('less predictable', xy=(7.3, 0.958), annotation_clip=False,fontsize=14,rotation='vertical')

ax.annotate('more predictable', xy=(7.3, 0.792), annotation_clip=False,fontsize=14,rotation='vertical')
#Посчитать энтропию для гостевых команд

entropy_means = matches.groupby(('season','away_team_api_id')).entropy.mean()

entropy_means = entropy_means.reset_index().pivot(index='season', columns='away_team_api_id', values='entropy')

entropy_means.columns = [teams[teams.team_api_id==x].team_long_name.values[0] for x in entropy_means.columns]

entropy_means = entropy_means[['Manchester United', 'Tottenham Hotspur', 'Chelsea']]

entropy_means.head(6)
#График

ax = entropy_means.plot(figsize=(12,8),marker='o')



#Название

plt.title('Away Teams Predictability', fontsize=16)



plt.xticks(rotation=50)



#Установить цвета

colors = [x.get_color() for x in ax.get_lines()]

colors_mapping = dict(zip(leagues.id,colors))



#Удалить название оси Х

ax.set_xlabel('')



#Добавить условное обозначение

plt.legend(loc='lower left')



#Добавить название

ax.annotate('', xytext=(7.2, 1),xy=(7.2, 1.04),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('', xytext=(7.2, 0.93),xy=(7.2, 0.89),

            arrowprops=dict(facecolor='black',arrowstyle="->, head_length=.7, head_width=.3",linewidth=1), annotation_clip=False)



ax.annotate('less predictable', xy=(7.3, 1.03), annotation_clip=False,fontsize=14,rotation='vertical')

ax.annotate('more predictable', xy=(7.3, 0.925), annotation_clip=False,fontsize=14,rotation='vertical')