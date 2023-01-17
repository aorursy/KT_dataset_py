#Imports

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tabulate import tabulate

import warnings

warnings.filterwarnings('ignore')

# load data

con = sqlite3.connect("../input/database.sqlite")

players = pd.read_sql_query("SELECT pl.player_fifa_api_id, pl.player_name, pl.height, pl.weight, p.overall_rating, p.preferred_foot, p.ball_control, p.balance, p.strength, p.stamina, p.aggression, MAX(p.date) FROM Player AS pl INNER JOIN Player_Attributes  AS p ON pl.player_fifa_api_id = p.player_fifa_api_id GROUP BY p.player_fifa_api_id ", con)

print(players.head())

print(players.info())
#Understanding Individual Features

#Height

plt.hist(players['height'])

plt.axvline(players['height'].mean(), color='r', linestyle='dashed', linewidth=2)

plt.legend([players['height'].mean()])

plt.xlabel('Height (cm)', fontsize=20)

plt.show()

#Weight

plt.hist(players['weight'])

plt.axvline(players['weight'].mean(), color='r', linestyle='dashed', linewidth=2)

plt.legend([players['weight'].mean()])

plt.xlabel('Weight (pound)', fontsize=20)

plt.show()

#Fifa Rating

plt.hist(players['overall_rating'])

plt.axvline(players['overall_rating'].mean(), color='r', linestyle='dashed', linewidth=2)

plt.legend([players['overall_rating'].mean()])

plt.xlabel('Rating', fontsize=20)

plt.show()
## FInding Relation of Height/Weight with Player Attributes

# Height and Weight Vs Balance

plt.scatter(players['height'],players['balance'])

plt.xlabel('Height (cm)', fontsize=20)

plt.ylabel('Balance', fontsize=20)

plt.title('Height Vs Balance',fontsize=20)

plt.show()

plt.scatter(players['weight'],players['balance'])

plt.xlabel('Weight (pound)', fontsize=20)

plt.ylabel('Balance', fontsize=20)

plt.title('Weight Vs Balance',fontsize=20)

plt.show()
#Height and Weight Vs Ball Control

plt.scatter(players['height'],players['ball_control'])

plt.xlabel('Height (cm)', fontsize=20)

plt.ylabel('Ball Control', fontsize=20)

plt.title('Height Vs Ball Control',fontsize=20)

plt.show()

plt.scatter(players['weight'],players['ball_control'])

plt.xlabel('Weight (pound)', fontsize=20)

plt.ylabel('Ball Control', fontsize=20)

plt.title('Weight Vs Ball Control',fontsize=20)

plt.show()
#Height and Weight Vs Stamina

plt.scatter(players['height'],players['stamina'])

plt.xlabel('Height (cm)', fontsize=20)

plt.ylabel('Stamina', fontsize=20)

plt.title('Height Vs Stamina',fontsize=20)

plt.show()

plt.scatter(players['weight'],players['stamina'])

plt.xlabel('Weight (pound)', fontsize=20)

plt.ylabel('Stamina', fontsize=20)

plt.title('Weight Vs Stamina',fontsize=20)

plt.show()
#Height and Weight Vs Strength

plt.scatter(players['height'],players['strength'])

plt.xlabel('Height (cm)', fontsize=20)

plt.ylabel('Strength', fontsize=20)

plt.title('Height Vs Strength',fontsize=20)

plt.show()

plt.scatter(players['weight'],players['strength'])

plt.xlabel('Weight (pound)', fontsize=20)

plt.ylabel('Strength', fontsize=20)

plt.title('Weight Vs Strength',fontsize=20)

plt.show()


#Height and Weight Vs Aggression

plt.scatter(players['height'],players['aggression'])

plt.xlabel('Height (cm)', fontsize=20)

plt.ylabel('Aggression', fontsize=20)

plt.title('Height Vs Aggression',fontsize=20)

plt.show()

plt.scatter(players['weight'],players['aggression'])

plt.xlabel('Weight (pound)', fontsize=20)

plt.ylabel('Aggression', fontsize=20)

plt.title('Weight Vs Aggression',fontsize=20)

plt.show()

# 1 for Right Footed and 0 for left footed

players['preferred_foot'] = np.where(players['preferred_foot']=='right', 1, 0)

labels =['Right Footed','Left Footed']

plt.pie(players.preferred_foot.value_counts(),labels=labels,autopct='%.2f')

plt.show()

#Difference in Left Footed and Right Footed Players

lefty = players[players["preferred_foot"] == 0]

righty = players[players["preferred_foot"] == 1]

print(tabulate([['Left Footed',np.mean(lefty['balance']),np.mean(lefty['ball_control']),np.mean(lefty['overall_rating'])], ['Right Footed',np.mean(righty['balance']),np.mean(righty['ball_control']),np.mean(righty['overall_rating'])]], headers=['Preferred Foot','Balance','Ball Control','Fifa Rating']))