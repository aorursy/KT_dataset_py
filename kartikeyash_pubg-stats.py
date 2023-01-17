# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

INPUT_DIR = "../input/"

# Any results you write to the current directory are saved as output.
data = pd.read_csv(INPUT_DIR + "train_V2.csv", nrows=10000)

data.info()
%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# create co relation matrix using seaborn heatmap

columns_to_ignore = ["Id", "groupId", "matchId"]

columns_to_show = [column for column in data.columns if column not in columns_to_ignore]

co_relation_matrix = data[columns_to_show].corr()

plt.figure(figsize=(11,9))

sns.heatmap(co_relation_matrix,

             xticklabels=co_relation_matrix.columns.values,

             yticklabels=co_relation_matrix.columns.values,

             linecolor="white",

             linewidth=0.1,

             cmap="RdBu"

           )

agg = data.groupby("groupId").size().to_frame('players_in_team')

data = data.merge(agg, how="left", on="groupId")



data['headshotKillsOverKills'] = data['headshotKills'] / data['kills']

data['headshotKillsOverKills'].fillna(0, inplace=True)



data['killPlaceOverMaxPlace'] = data['killPlace'] / data['maxPlace']

data['killPlaceOverMaxPlace'].fillna(0, inplace=True)

data['killPlaceOverMaxPlace'].replace(np.inf, 0, inplace=True)



corr = data[['killPlace', 'walkDistance', 'headshotKillsOverKills', 'players_in_team',

             'killPlaceOverMaxPlace', 'winPlacePerc']].corr()

sns.heatmap(corr,

    xticklabels=corr.columns.values,

    yticklabels=corr.columns.values,

    annot=True,

    linecolor='white',

    linewidth=0.1,

    cmap="RdBu"

)