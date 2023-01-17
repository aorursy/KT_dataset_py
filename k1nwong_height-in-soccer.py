import matplotlib.pyplot as plt

import numpy as np

import sqlite3

import pandas as pd

import os.path

%matplotlib inline
#dark theme for matplotlib for jupyter notebook

# from jupyterthemes import jtplot

# jtplot.style(theme='onedork')
#import table in the database 

cnx = sqlite3.connect('C:/Users/chiu/Documents/UCSD Python/Week5-Visualization/database.sqlite')

# country = pd.read_sql_query("SELECT * FROM Country", cnx)

# league = pd.read_sql_query("SELECT * FROM League", cnx)

# match = pd.read_sql_query("SELECT * FROM Match", cnx)

player = pd.read_sql_query("SELECT * FROM Player", cnx)

player_att = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

# team = pd.read_sql_query("SELECT * FROM Team", cnx)

# team_att = pd.read_sql_query("SELECT * FROM Team_Attributes", cnx)
player_att.columns

player.columns
player_att[['overall_rating', 'heading_accuracy', 'agility', 'ball_control']].describe()

boxplot = player_att.boxplot(['overall_rating', 'heading_accuracy', 'agility', 'ball_control'])
avg_stat = player_att.groupby('player_fifa_api_id').mean()

print("Average stat's shape:", avg_stat.shape) #11062

print("Number of players:", player.shape) #11060

player_avg_stat = player.merge(avg_stat, on='player_fifa_api_id', how='inner')

print("Number of rows in the combined table:", player_avg_stat.shape) #11056   rows, removed all the rows with player id that does not in common with the other table
player_avg_stat[['player_fifa_api_id','player_name','height','ball_control','heading_accuracy']].head()
player_avg_stat.columns
player_avg_stat['height'][:5]
print(np.corrcoef(player_avg_stat['height'],player_avg_stat['heading_accuracy']))

fig, axis = plt.subplots(figsize=(10,10))

axis.yaxis.grid(False)

axis.set_title("Player's Height vs Heading Accuracy ",fontsize=25)

axis.set_xlabel("Player's Height",fontsize=20)

axis.set_ylabel("Heading Accuracy",fontsize=20)

X = player_avg_stat['height']

Y = player_avg_stat['heading_accuracy']

axis.scatter(X, Y)

# plt.xticks(player_avg_stat['height'])

plt.show()

plt.scatter(player_avg_stat['ball_control'],player_avg_stat['dribbling'])

#the graph shows a positive relationship

np.corrcoef(player_avg_stat['ball_control'],player_avg_stat['dribbling'])

#the matrix shows a really high correlation.
from pandas.plotting import scatter_matrix

# scatter_matrix(player_avg_stat[['agility','balance','ball_control','dribbling']])
print(np.corrcoef(player_avg_stat['height'],player_avg_stat['ball_control']))

fig, axis = plt.subplots(figsize=(10,10))

axis.yaxis.grid(False)

axis.set_title("Player's Height vs Ball Control ",fontsize=25)

axis.set_xlabel("Player's Height",fontsize=20)

axis.set_ylabel("Ball Control",fontsize=20)

X = player_avg_stat['height']

Y = player_avg_stat['ball_control']

axis.scatter(X, Y)

# plt.xticks(player_avg_stat['height'])

plt.show()
