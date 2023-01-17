import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Read the input files

playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

inj = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

trk = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')



def change_width(ax, new_value) :

    for patch in ax.patches :

        current_width = patch.get_width()

        diff = current_width - new_value



        # we change the bar width

        patch.set_width(new_value)



        # we recenter the bar

        patch.set_x(patch.get_x() + diff * .5)
f,ax=plt.subplots(figsize=(13,8))

sns.countplot(inj['BodyPart'],palette=sns.color_palette("Blues_d"))



change_width(ax,.55)
sns.countplot(y=inj['Surface'],palette=sns.color_palette("Blues_d"))
f,ax=plt.subplots(figsize=(16,8))

sns.countplot(x=inj['Surface'],hue=inj['BodyPart'])
d = inj.iloc[:,5:]



d=d.sum(axis=1)

d.sort_values(inplace=True)

d.replace(1,'DM_M1',inplace=True)

d.replace(2,'DM_M7',inplace=True)

d.replace(3,'DM_M28',inplace=True)

d.replace(4,'DM_M42',inplace=True)



f,ax=plt.subplots(figsize=(12,8))

sns.countplot(x=inj['Surface'],hue=d)
f,ax=plt.subplots(figsize=(12,8))

sns.countplot(x=inj['BodyPart'],hue=d)