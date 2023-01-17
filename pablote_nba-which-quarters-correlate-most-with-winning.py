# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Loosely forked off of kernel "garyxcheng/leading-in-the-first-half-winning'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

pd.options.mode.chained_assignment = None
df16 = pd.read_csv("../input/2016-17_teamBoxScore.csv")
df17 = pd.read_csv("../input/2017-18_teamBoxScore.csv")
# Choose one of the following options to pick which dataset
# df = df16
df = df17
#df = pd.concat((df16, df17), ignore_index = True)

df = df[["teamAbbr", "teamLoc", "teamPTS", "teamPTS1", "teamPTS2", "teamPTS3", "teamPTS4", "teamPTS5", 
         "opptAbbr", "opptLoc", "opptPTS", "opptPTS1", "opptPTS2", "opptPTS3", "opptPTS4", "opptPTS5"]]

# Each pair of rows is equivalent, but reversing the first and second teams.
# Remove each second row to avoid this "duplicated" (mirrored) data.

## This comment is correct, but the technique used to remove the duplicate rows needs examination.
## The reason that there are 2 rows per game is to give the ability to search for games from a team's perspective.
## I will give some examples below on some different techniques for eliminating duplicate data.

for i in range(1, 6):
  df["ptsDiff%s" % i] = df["teamPTS%s" % i] - df["opptPTS%s" % i]
df["teamWin"] = (df["teamPTS"] > df["opptPTS"]).apply(lambda x: x and 1.0 or -1.0)
df["wentToOT"] = (df["teamPTS5"] + df["opptPTS5"]) > 0

#print (df.shape)
#print (df.loc[0, :])
## Let's start by including all rows
# df = df.iloc[::2, :].reset_index(drop = True)

## We can eliminate overtime games for now to reduce complexity.
dfReg = df[["teamAbbr", "teamLoc", "ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin", "wentToOT"]]
dfReg = dfReg.drop(dfReg[(dfReg.wentToOT == True)].index)
dfReg.drop("wentToOT", axis=1, inplace=True)
print (dfReg.shape)
print (dfReg.corr())
print ("")

## Sum team points per quarter and team win​.
sumsReg = dfReg.select_dtypes(pd.np.number).sum().rename('total')
sumsReg = sumsReg[["ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin"]]
print (sumsReg)
## Let's try looking from the perspective of Utah Jazz games ending during regulation.
dfJazz = dfReg[["teamAbbr", "ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin"]]
dfJazz.drop(dfJazz[(dfJazz.teamAbbr != "UTA")].index, inplace=True)
dfJazz.drop(["teamAbbr"], axis=1, inplace=True)
print (dfJazz.shape)
print (dfJazz.corr())
print ("")

## Sum team points per quarter and team win.
sumsJazz = dfJazz.select_dtypes(pd.np.number).sum().rename('total')
sumsJazz = sumsJazz[["ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin"]]
print (sumsJazz)
## Let's try looking from the perspective Utah Jazz home games ending during regulation.
dfJazzHome = dfReg[["teamAbbr", "teamLoc", "ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin"]]
dfJazzHome.drop(dfJazzHome[(dfJazzHome.teamLoc == "Away")].index, inplace=True)
dfJazzHome.drop(dfJazzHome[(dfJazzHome.teamAbbr != "UTA")].index, inplace=True)
dfJazzHome.drop(["teamAbbr", "teamLoc"], axis=1, inplace=True)
print (dfJazzHome.shape)
print (dfJazzHome.corr())
print ("")

## Sum team points per quarter and team win
sumsJazzHome = dfJazzHome.select_dtypes(pd.np.number).sum().rename('total')
sumsJazzHome = sumsJazzHome[["ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin"]]
print (sumsJazzHome)
## Focusing on Utah Jazz home games, we have a small enough sample to visualize the correlation.
## Correlation between periods 1 and 2
import numpy as np
from numpy.polynomial.polynomial import polyfit

x = dfJazzHome['ptsDiff1']
y = dfJazzHome['ptsDiff2']

b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')

plt.scatter(x,y,color='cyan')
plt.show()
## Correlation between periods 3 and 4
x = dfJazzHome['ptsDiff3']
y = dfJazzHome['ptsDiff4']

b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')

plt.scatter(x,y,color='cyan')
plt.show()
## Correlation between period 4 and game outcome
x = dfJazzHome['ptsDiff4']
y = dfJazzHome['teamWin']

b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')

plt.scatter(x,y,color='cyan')
plt.show()