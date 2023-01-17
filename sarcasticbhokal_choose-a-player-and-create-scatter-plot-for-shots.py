import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy as scp
db = pd.read_csv('../input/shot_logs.csv')



#########################

# Choose your player here. 

#########################



Playername='stephen curry'
db.head(5)
dfPlayerOnly=db[['player_name','FGM','SHOT_DIST','CLOSE_DEF_DIST']]
dfPlayerOnly.head(5)
dfPlayer=dfPlayerOnly[ dfPlayerOnly['player_name'] == Playername]
dfPlayer['SHOT_DIST'].max() 

# This is a check to ensure that the chosen player is in the csv file.

# If this returns Nan, either the player is not in the csv file or there is a spelling error in your playername string.

import pylab

dfPlayerMade=dfPlayer[dfPlayer['FGM']==1]

dfPlayerMissed=dfPlayer[dfPlayer['FGM']==0]
made=plt.scatter(dfPlayerMade['SHOT_DIST'],dfPlayerMade['CLOSE_DEF_DIST'],c='b')

missed=plt.scatter(dfPlayerMissed['SHOT_DIST'],dfPlayerMissed['CLOSE_DEF_DIST'],marker='.',c='r')

plt.xlabel("SHOT DISTANCE")

plt.ylabel("CLOSEST DEFENDER DISTANCE")

plt.title(Playername + " made-missed scatterplot")

plt.xlim(-2)

plt.ylim(-2)

plt.legend((made,missed),('made','missed'),scatterpoints=1)