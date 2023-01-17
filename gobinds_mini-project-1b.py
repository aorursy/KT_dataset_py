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
#add in matplotlib library so we can create visualizations.
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

players = pd.read_csv('../input/nba-players-data/all_seasons.csv') #set dataframe of csv as "players"
players.tail(5) #preview the end of the dataset to see attributes
age=players[['age']]   #define age and points per game in dataframe
pts=players[['pts']]

plt.style.use('seaborn-poster') #set style to seaborn-poster so that plots are large and easier to read
plt.style.use('fivethirtyeight') #set style to fivethirtyeight so that there are guidelines

plt.scatter(age,pts, alpha=0.5, c='blue') #scatter plot showing age on the x-axis and pts on the y.
kobe = players[players.player_name == 'Kobe Bryant'] #create new dataframe with the values of kobe
kobe #preview the new kobe dataframe
kb_age=kobe[['age']] #define age and points per game column in kobe dataframe
kb_pts=kobe[['pts']]

plt.scatter(kb_age,kb_pts, alpha=0.5, c='purple')  #scatterplot using kobe attributes
#Here we will set "season" as a new dataframe that makes season, the y axis, and averages all of the stats in the dataframe.
season = players.groupby(['season']).mean()

season['pts'].plot(kind='barh',legend=True)   #plot points per season in a horizontal bar graph using the season dataframe we just created.

height=players['player_height']/30.48  #define height in dataframe and convert cm to feet
weight=players['player_weight']*2.205  #define weight and convert kg to lb

plt.scatter(height,pts, alpha=0.5, c='blue') #scatterplot using height as x-axis and points as y-axis
plt.scatter(weight,pts, alpha=0.5, c='red')  #scatterplot using height as x-axis and points as y-axis
draft_players= pd.read_csv('../input/nba-players-data/all_seasons.csv') #create new df to manipulate

#creating new column with only drafted players. We want to make sure these are only storing the drafted years and not undrafted.
draft_players["drafted"]= draft_players["draft_year"].str.isdigit() 

#creating an object that will store drafted players
was_drafted = draft_players['drafted'] == True

#creating a filter that will only include drafted players
draft_players.where(was_drafted, inplace = True) #where function finds players that are drafted

after_95 = draft_players["draft_year"].astype(float)>=1996  #this will only find players that were drafted 1996 and later
draft_players.where(after_95, inplace = True) 

#set "draft_class" as a new dataframe that makes draft_year, the row axis, and averages all of the stats in the dataframe.
draft_class = draft_players.groupby(['draft_year']).mean()



draft_class.plot.bar(y=['pts','reb','ast']) #plot a segmented bar graph containing points, rebounds, and assists from the draft_class dataframe
#set "country" as a new dataframe that makes country, the row axis, and averages all of the stats in the dataframe.
country = players.groupby(['country']).mean()

country.plot.bar(y=['pts','reb','ast']) #plot a segmented bar graph containing points, rebounds, and assists from the country dataframe