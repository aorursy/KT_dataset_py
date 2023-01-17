# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

import pandas_profiling 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
stats = pd.read_csv("../input/stats.csv")
stats.head()
stats.info()
plt.figure(figsize=(16,6))

Equipos = stats['team'].value_counts()

plt.title('Más participaciones en la liga desde el 2006')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="Salmon", orient="v")
continents = ['Tottenham Hotspur','Manchester United', 'Liverpool', 'Manchester City', 'Everton', 'Arsenal', 'Chelsea']

stats1 = stats[stats.team.isin(continents)]
pandas_profiling.ProfileReport(stats1) 
plt.figure(figsize=(16,6))

Equipos =  stats1.groupby('team')['wins'].sum()

plt.title('Más victorias en liga')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="seagreen")
plt.figure(figsize=(16,6))

Equipos =  stats1.groupby('team')['goals'].sum()

plt.title('Más goles anotados en liga')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="lightgreen")
plt.figure(figsize=(16,6))

Equipos =  stats1.groupby('team')['goals_conceded'].sum()

plt.title('Menos goles recibidos')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="orange")
plt.figure(figsize=(16,6))

Equipos =  stats1.groupby('team')['total_pass'].sum()

plt.title('Con más pases en liga')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="silver")
plt.figure(figsize=(16,6))

Equipos =  stats1.groupby('team')['penalty_conceded'].sum()

plt.title('Con más penales en contra')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="slateblue")
plt.figure(figsize=(16,6))

Equipos =  stats1.groupby('team')['own_goals'].sum()

plt.title('Con más goles en contra')

sns.barplot(x=Equipos[:10].keys(), y=Equipos[:10].values, color="yellow")