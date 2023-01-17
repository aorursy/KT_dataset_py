# Import necessary packages

import os

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import datetime 



# Handle date time conversions between pandas and matplotlib

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



# Use white grid plot background from seaborn

sns.set(font_scale=1.5, style="whitegrid")
player = pd.read_csv("../input/nfl-big-data-bowl-2021/players.csv")

player.info()
player['height'].value_counts()
player['height'] = player['height'].str.replace('[-]', '') 
#convert the height column from object type to int 

player['height'] = pd.to_numeric(player['height'])



player.head(10)
#sort the dataframe values from the height column 

player = player.sort_values(by=['weight'], ascending=False)

#convert birthdate column to age

#player['birthDate'] = pd.to_datetime(player['birthDate'], format='%m%d%y')

def from_dob_to_age(born):

    today = datetime.date.today()

    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

player['birthDate'] =  pd.to_datetime(player['birthDate'], infer_datetime_format=True)

player['age'] = player['birthDate'].apply(lambda x: from_dob_to_age(x))

player.head(10)
player = player.sort_values(by=['age'], ascending=False)

player
player_30 = player[player['age'] > 30] 

player_30
player['position'].value_counts() 

table = player.pivot_table(index=['position'], aggfunc='median')

table.head(10) 
del table['nflId']

table.info()

table.hist()
table.sort_values(by='weight',ascending=False).plot(kind='bar' ,figsize =(15,6), title ='positions grouped by age & weight')
plays = pd.read_csv("../input/nfl-big-data-bowl-2021/plays.csv")

plays
plays['playDescription'].value_counts() 