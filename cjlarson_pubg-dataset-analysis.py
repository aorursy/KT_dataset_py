#Thanks to Kaggle for setting this up for me. 

#As a former PUBG player I thought this dataset would be worth taking a stab at and maybe producing some models over, but if it's anything like the game itself it might be a PiTA. Still should be fun though.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # pretty picture helper
import statsmodels.api as sm #linear model helper


import os
print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")


#let's check out the top of the dataset and some summary statistics
df.head()
df.describe()
#it makes the most sense to me to view this data by the match, considering the outcome of the match is what we are predicting. 
#This will give us the first 5 matches in the data set. It's also easier to chop this down because looking at nearly 50,000 matches is going to be expensve to looka t processing-wise

dfsnip = df.loc[df['matchId'] < 5]

dfsnip.head()

dfsnip.describe()
#Looking at the above tells us there are 475 seperate Id's in the 5 games. This sounds about right considering that there are up to 100 players per game. 
#We can also see that the min in this case is often 0. This makes sense because if you've ever played a game of PUBG, you've probably been murdered in the first 5 seconds before figuring out your surroundings 
#IOther notable stats may be that the max amount of kills in this subset of data is 11 (I am thinking the higher your kill count, the higher you will likely place in the game), the max weapons acquired is 14 (if you acquire this many weapons, you probably got good loot)

#Let's visualize the kills vs WinPlacePerc theory

viz1 = dfsnip.plot.scatter(x='kills',y='winPlacePerc',c='Blue')


#Seems legit, the winplace seems to go up as the kills increase, let's view the whole set

#viz2 = df.plot.scatter(x='kills',y='winPlacePerc',c='Red')

#this also somewhat holds the theory, but the immense amount of data muddles it a little bit.
#now let's look at weapons acquired vs wins


viz3 = dfsnip.plot.scatter(x='weaponsAcquired',y='winPlacePerc',c='Blue')
#viz4 = df.plot.scatter(x='weaponsAcquired',y='winPlacePerc',c='Red')
#This seems less promising with the large dataset again, but I think it could still hold up in some models. The sheer amount of data that is in this dataset could cause some breakdowns and may also be ripe for some outlier removals. 

#Now we can look at all correlations, so that we don't have to do this for every possible correlation:

dfsnip.corr()
#Interstingly we see a high correlation between walkDistance and winPlacePerc .... 
viz5 = dfsnip.plot.scatter(x='walkDistance',y='winPlacePerc',c='Blue')

#Which I guess makes sense because as you get towards the end of the game you are going to have to move to stay in the "circle", the barrier forcing the players to get closer together at the end of the game.
#Let's try a model just based off of the 3 variables discussed so far:

target = df[['winPlacePerc']]
start_features = df[['walkDistance','kills','weaponsAcquired']]
model = sm.OLS(target,start_features).fit()

model.summary()
#Our small base model yields an r^2 of 0.878! That's.... actually not that bad for a model with only 3 variables and not a whole ton of time put into it! We will be taking a look at how this model performs on the larger dataset in the future.