# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
football = pd.read_csv("../input/results.csv", sep = ',', header=0, index_col=0) 

football.head()
for team in football.home_team:

    print(team)
number_home_team = len(set(football.home_team))

unique_home_team = set(football.home_team)

#unique_home_team

# 293

#all_home_team = len(football.home_team)

# 40262
number_away_team = len(set(football.away_team))

unique_away_team = set(football.away_team)

#unique_away_team

number_away_team

# 291

all_away_team = len(football.away_team)

# 40262
print ("These teams have only played away: ",unique_away_team.difference(unique_home_team)) 

print ("These teams have only played at home:", unique_home_team.difference(unique_away_team)) 
AwayX=football.away_team.value_counts()

HomeX=football.home_team.value_counts()

#len(AwayX)
FirstAway=football.away_team.value_counts().head()

FirstHome=football.home_team.value_counts().head()
#(histX['Argentina'])
import matplotlib.pyplot as plt

plt.hist(AwayX)

plt.show()

plt.hist(HomeX)

plt.show()
fig, ax = plt.subplots()



# Example data





ax.barh(FirstAway.index, FirstAway)

ax.set_xlabel('Number of match')

ax.set_title('Top 5 most frequent away teams')



plt.show()
fig, ax = plt.subplots()



# Example data





ax.barh(FirstHome.index, FirstHome)

ax.set_xlabel('Number of match')

ax.set_title('Top 5 most frequent home teams')



plt.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



data = [go.Bar(

        y=FirstHome.index,

        x=FirstHome,

        orientation='h')]

fig = go.Figure(data=data)



py.offline.iplot(fig)

data = [go.Bar(

        y=FirstAway.index,

        x=FirstAway,

        orientation='h')]

fig = go.Figure(data=data)



py.offline.iplot(fig)
EnglandAway=football.away_team[England]
EnglandA = football[['away_team']].sum(axis=1).where(football['away_team'] == 'England').count()

EnglandH = football[['home_team']].sum(axis=1).where(football['home_team'] == 'England').count()



# Intitialise data of lists  

data = [{'Home': EnglandH, 'Away': EnglandA}] 

  

# Creates padas DataFrame by passing  

# Lists of dictionaries and row index. 

df = pd.DataFrame(data, index =['England']) 

  

# Print the data 

df 