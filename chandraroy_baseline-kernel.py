import numpy as np

import pandas as pd

import os as os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input/nfl-playing-surface-analytics/"))
playList= pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

playerTrackData= pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")

InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")
playList.columns
playerTrackData.columns
InjuryRecord.columns
playList.head(4)
playerTrackData.head(4)
InjuryRecord.head(4)
print(playList.shape)

print(playerTrackData.shape)

print(InjuryRecord.shape)
playList.isna().sum()
playerTrackData.isna().sum()
InjuryRecord.isna().sum()
pp = sns.pairplot(playList, hue="StadiumType")
sns.pairplot(playList, hue="RosterPosition")
sns.pairplot(playList, hue="Weather")
sns.pairplot(playList, hue="Position")
pp3 = sns.pairplot(InjuryRecord)