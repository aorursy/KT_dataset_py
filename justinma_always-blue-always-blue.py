import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("../input/LeagueofLegends.csv")
grp = data.set_index(['League','Season','Year'])
grp = grp.drop(['blueTeamTag','redTeamTag'],axis=1)

grp.drop('MatchHistory',inplace=True,axis=1)
NA_Spring = grp.xs(['North_America','Spring_Season'])

NA_Summer = grp.xs(['North_America','Summer_Season'])

EU_Spring = grp.xs(['Europe','Spring_Season'])

EU_Summer = grp.xs(['Europe','Summer_Season'])
blue = EU_Spring.groupby('gamelength').bResult.sum()

red = EU_Spring.groupby('gamelength').rResult.sum()

sns.plt.plot(blue.index,blue)

sns.plt.plot(red.index,red,color='r')

plt.title('Euro Spring Season')

print("Blue Win {} , Red Win {}".format(NA_Summer.bResult.mean()*100 ,NA_Summer.rResult.mean()*100))
blue = EU_Summer.groupby('gamelength').bResult.sum()

red = EU_Summer.groupby('gamelength').rResult.sum()

sns.plt.plot(blue.index,blue)

sns.plt.plot(red.index,red,color='r')

plt.title('Euro Summer Season')

print("Blue Win {} , Red Win {}".format(NA_Summer.bResult.mean()*100 ,NA_Summer.rResult.mean()*100))
blue = NA_Summer.groupby('gamelength').bResult.sum()

red = NA_Summer.groupby('gamelength').rResult.sum()

sns.plt.plot(blue.index,blue)

sns.plt.plot(red.index,red,color='r')

plt.title('North America Summer Season')

print("Blue Win {} , Red Win {}".format(NA_Summer.bResult.mean()*100 ,NA_Summer.rResult.mean()*100))
blue = NA_Spring.groupby('gamelength').bResult.sum()

red = NA_Spring.groupby('gamelength').rResult.sum()

sns.plt.plot(blue.index,blue)

sns.plt.plot(red.index,red,color='r')

plt.title('North America Spring Season')

print("Blue Win {} , Red Win {}".format(NA_Spring.bResult.mean()*100 ,NA_Spring.rResult.mean()*100))