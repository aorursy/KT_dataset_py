# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import plotly library

import plotly.express as px

import plotly.io as pio



# Dataframe with Drivers Championship

df_drivers = pd.read_csv("../input/fia-f1-19502019-data/drivers_championship_1950-2020.csv")



# Dataframe with Race Results

df_results = pd.read_csv("../input/fia-f1-19502019-data/race_results_1950-2020.csv")



# Dataframe with Race Wins

df_wins = pd.read_csv("../input/fia-f1-19502019-data/race_wins_1950-2020.csv")



# Dataframe with Fastest laps

df_laps = pd.read_csv("../input/fia-f1-19502019-data/fastest_laps_1950-2020.csv")



# Let's rename column 'Name' to 'Driver'

df_drivers = df_drivers.rename(columns={'Name': 'Driver'})

df_results = df_results.rename(columns={'Name': 'Driver'})

df_wins = df_wins.rename(columns={'Name': 'Driver'})

df_laps = df_laps.rename(columns={'Name': 'Driver'})



# Let's replace Raikkonen name

df_drivers = df_drivers.replace(['Kimi RÃ¤ikkÃ¶nen'],['Kimi Räikkönen'])

df_results = df_results.replace(['Kimi RÃ¤ikkÃ¶nen'],['Kimi Räikkönen'])

df_wins = df_wins.replace(['Kimi RÃ¤ikkÃ¶nen'],['Kimi Räikkönen'])

df_laps = df_laps.replace(['Kimi RÃ¤ikkÃ¶nen'],['Kimi Räikkönen'])
# Count races by driver

races = df_results.groupby('Driver')['Driver'].count()

races = pd.DataFrame(races)

races.columns = ['Races']

races.reset_index(level=0, inplace=True)



# Sort Drivers by nº of races

races.sort_values(by=['Races'], inplace=True, ascending=False)



# Let's take the top 20

races_20 = races.head(20)

races_20 = races_20[::-1]



# Let's select the charts template

pio.templates.default = "plotly_dark"



# Plot chart

fig = px.bar(races_20, x='Races', y='Driver',color='Races',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with The Most Races','y':0.95,'x':0.5})

fig.show()
# Let's drop data from 2020 championship

champions = df_drivers.drop(df_drivers[df_drivers.Year == 2020].index)



# Let's drop all drivers who weren't champions

champions = champions.drop(champions[champions.Position != '1'].index)



# Count Championships by driver

champs = champions.groupby('Driver')['Driver'].count()

champs = pd.DataFrame(champs)

champs.columns = ['Championships']

champs.reset_index(level=0, inplace=True)



# Sort winners

champs.sort_values(by=['Championships'], inplace=True, ascending=False)



# Let's take the top 20

champs = champs.head(20)

champs = champs[::-1]



# Plot chart

fig = px.bar(champs, x='Championships', y='Driver',color='Championships',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with The Most Championships','y':0.95,'x':0.5})

fig.show()
# Count wins by driver

wins = df_wins.groupby('Driver')['Driver'].count()

wins = pd.DataFrame(wins)

wins.columns = ['Wins']

wins.reset_index(level=0, inplace=True)



# Sort winners

wins.sort_values(by=['Wins'], inplace=True, ascending=False)



# Let's take the top 20

wins_20 = wins.head(20)

wins_20 = wins_20[::-1]



# Plot chart

fig = px.bar(wins_20, x='Wins', y='Driver',color='Wins',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with The Most Wins','y':0.95,'x':0.5})

fig.show()
# Create column for podium

conditions = [(df_results['Position'] == '1') | (df_results['Position'] == '2') | (df_results['Position'] == '3')]

values = ['Podium']

df_results['Podium'] = np.select(conditions, values, default=0)



# Let's create a new dataframe

results = df_results.copy()



# Let's drop drivers who weren't at the podium

results = results.drop(results[results.Podium != 'Podium'].index)



# Sum podiums

podiums = results.groupby('Driver')['Driver'].count()

podiums = pd.DataFrame(podiums)

podiums.columns = ['Podiums']

podiums.reset_index(level=0, inplace=True)

podiums.sort_values(by=['Podiums'], inplace=True, ascending=False)



# Let's take the top 20

podiums_20 = podiums.head(20)

podiums_20 = podiums_20[::-1]



# Plot chart

fig = px.bar(podiums_20, x='Podiums', y='Driver',color='Podiums',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with The Most Podiums','y':0.95,'x':0.5})

fig.show()
# Count fast laps by drivers

laps = df_laps.groupby('Driver')['Driver'].count()

laps = pd.DataFrame(laps)

laps.columns = ['Laps']

laps.reset_index(level=0, inplace=True)

laps.sort_values(by=['Laps'], inplace=True, ascending=False)



# Let's take the top 20

laps_20 = laps.head(20)

laps_20 = laps_20[::-1]



# Plot chart

fig = px.bar(laps_20, x='Laps', y='Driver',color='Laps',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with The Most Fast Laps','y':0.95,'x':0.5})

fig.show()
# Merge races and podiums dataframes

ratio_podium_race = races.merge(podiums, how='left', on='Driver')



# Create a column for the ratio

ratio_podium_race['Ratio'] = round(100*ratio_podium_race['Podiums']/ratio_podium_race['Races'],1)



# Sort by ratio

ratio_podium_race.sort_values(by=['Ratio'], inplace=True, ascending=False)



# Drop drivers with less than 30 races

ratio_podium_race = ratio_podium_race[ratio_podium_race['Races']>30]



# Let's take the top 20

ratio_podium_race_20 = ratio_podium_race.head(20)

ratio_podium_race_20 = ratio_podium_race_20[::-1]



# Plot chart

fig = px.bar(ratio_podium_race_20, x='Ratio', y='Driver',color='Ratio',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with Largest Podiums/Races Ratio','y':0.95,'x':0.5})

fig.show()
# Merge races and wins dataframes

ratio_wins_race = races.merge(wins, how='left', on='Driver')



# Create a column for the ratio

ratio_wins_race['Ratio'] = round(100*ratio_wins_race['Wins']/ratio_wins_race['Races'],1)



# Sort drivers by ratio

ratio_wins_race.sort_values(by=['Ratio'], inplace=True, ascending=False)



# Drop drivers with less than 30 races

ratio_wins_race = ratio_wins_race[ratio_wins_race['Races']>30]



# Let's take the top 20 

ratio_wins_race_20 = ratio_wins_race.head(20)

ratio_wins_race_20 = ratio_wins_race_20[::-1]



# plot chart

fig = px.bar(ratio_wins_race_20, x='Ratio', y='Driver',color='Ratio',width=600, height=500)

fig.update_layout(title={'text': 'Drivers with Largest Wins/Races Ratio','y':0.95,'x':0.5})

fig.show()