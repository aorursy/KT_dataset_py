# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import matplotlib.cm as cm



#------------------------------------------------------------------

# Functions used in this exercise

# Note: There is probably much better way in separating the player name and country from player.

# Since the primary objective is to analyze the data, I'm compromising the elegancy 

def get_player_name( player):

    vals = re.findall(r'\((.*?)\)', player)

    name = player

    for v in vals:

        to_rep = '(' + v + ')'

        name = name.replace(to_rep, '')



    return name



def get_country( player):

    vals = re.findall(r'\((.*?)\)', player)

    country = vals[-1]

    if 'ICC' in country and len(country.split('/')) == 2: # If a player played for his country and ICC, then ignore ICC

        country = country.replace('ICC','').replace('/','')

    return country

#------------------------------------------------------------------



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Read the data in

print ("Reading the data...")

df = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv', encoding='ISO-8859-1')

print ("Done")



# preview the data

df.head()



# Drop the player profile column.. No use



df.drop('Player Profile', axis=1, inplace=True)



# Player contains the player name and country

# Create 2 new columns with just player name and country



df['Name'] = df['Player'].apply(get_player_name)

df['Country'] = df['Player'].apply(get_country)



# Look at the number of players by country

df.groupby('Country').count()



# hhmm there are some players played for multiple countries

# Lets add another column to store the number of countries

df['NumCountries'] = df['Country'].apply(lambda x: len(x.split('/')))



# Did someone play for more than 2 countries?

print ("Number of Countries")

print (df.NumCountries.value_counts())
# Who played for more than 1 country?

print (df.loc[ df['NumCountries'] > 1 , ['Player', 'Mat'] ])



# Out of 3001 players, 15 played for multiple countries. That is 0.5% Small number.. Delete these players

df.drop( df[ df['NumCountries'] > 1 ].index, inplace=True )
# Lets do some charting



# How many players per country?



players_by_country =  df.groupby('Country')['Player'].count()



plt.xticks(rotation='vertical')

plt.bar(x = players_by_country.index,height=players_by_country.values)

plt.show()
# Look at the column types

df.dtypes
# Remove * in HS. This indicates the batsman was not-out..

df['HS'] = df['HS'].str.replace('*','')



# Inn, NO, Runs, HS, Avg, 100, 50, 0 are object.. Convert them to numeric... Some players have not scored any runs or does not

# have avergage.. Convert them to NaN using 'coerce'



for col in ['Inn', 'NO', 'Runs', 'Avg', 'HS', '100', '50', '0']:

    df[col] = pd.to_numeric(df[col], errors='coerce')



    

# Now look at the types

df.dtypes

df
# Span contains the range of the year in which the player played

# Create new columns From/To store the debut year and retired/finally dropped from the team year



df = pd.concat(

    [

        df,

        df['Span'].apply(

            lambda x:

                pd.Series(

                    {

                        'From' : int(x.split('-')[0]),

                        'To' : int(x.split('-')[1])

                    }

                )

        )

    ],

    axis=1)



df.head()

# Create a column to store the number of years the player was active

df['SpanYears'] = df['To'] - df['From']



df.head()
# Which player had longest career



print ("Player with longest career")



# Using this approach instead of idxmax so we can identify if there are more than 1 player with long career

df[

    ['Player', 'Span']

][

    df['SpanYears'] == df['SpanYears'].max()

]
# Who had more number of ducks (0)



print ("Player with most ducks")

df[

    ['Player', 'Inn', 'Runs', '0']

][

    df['0'] == df['0'].max()

]



# No surprise who that player is.. But he is a great bowler and gentleman though
# Who coverts 50s to 100s more often

# For this create a new data frame which has players who scored more than 1000 runs

# to avoid cases where tail enders or players who have not played many matches scoring 1 or 2 100s by luck





gp = df.drop( df[ df['Runs'] < 1000 ].index )

gp['100To50'] = gp['100'] / gp['50']



print ("Player who converts more 50s to 100s")

gp[

    ['Player', 'Inn', 'Runs', '100', '50', '100To50']

][

    gp['100To50'] == gp['100To50'].max()

].sort_values('Runs', ascending=True)