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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

pd.options.mode.chained_assignment = None
df16 = pd.read_csv("../input/2016-17_teamBoxScore.csv")
df17 = pd.read_csv("../input/2017-18_teamBoxScore.csv")
# Choose one of the following options to pick which dataset
# df = df16
# df = df17
df = pd.concat((df16, df17), ignore_index = True)

df = df[["teamAbbr", "teamPTS", "teamPTS1", "teamPTS2", "teamPTS3", "teamPTS4", "teamPTS5", "opptAbbr", "opptPTS", "opptPTS1", "opptPTS2", "opptPTS3", "opptPTS4", "opptPTS5"]]

# Each pair of rows is equivalent, but reversing the first and second teams.
# Remove each second row to avoid this "duplicated" (mirrored) data.
df = df.iloc[::2, :].reset_index(drop = True)
for i in range(1, 6):
  df["ptsDiff%s" % i] = df["teamPTS%s" % i] - df["opptPTS%s" % i]
df["teamWin"] = (df["teamPTS"] > df["opptPTS"]).apply(lambda x: x and 1.0 or -1.0)
df["wentToOT"] = (df["teamPTS5"] + df["opptPTS5"]) > 0

# print (df.shape)
# print (df.loc[0, :])
# Correlations for all games.

df2 = df[["ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "teamWin"]]
print (df2.corr())
# Check the correlations for games that ended up going to overtime.

dfOT = df[["ptsDiff1", "ptsDiff2", "ptsDiff3", "ptsDiff4", "ptsDiff5", "teamWin", "wentToOT"]]
dfOT = dfOT[dfOT["wentToOT"]].drop(["wentToOT"], axis=1)
print (dfOT.corr())