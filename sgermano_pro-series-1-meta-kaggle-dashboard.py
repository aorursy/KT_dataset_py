# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data_competitions = pd.read_csv('../input/Competitions.csv');
data_compTags = pd.read_csv('../input/CompetitionTags.csv');
data_tags = pd.read_csv('../input/Tags.csv');
columns = ['Id', 'Title', 'NumScoredSubmissions', 'RewardType', 'RewardQuantity', 'TotalCompetitors', 'TotalSubmissions'];
df_comp = pd.DataFrame(data_competitions, columns=columns);
df_compTags = pd.DataFrame(data_compTags);
df_tags = pd.DataFrame(data_tags);

df_data = pd.merge(df_comp, df_compTags, left_on="Id", right_on="CompetitionId", how="left");
df_data = pd.merge(df_data, df_tags.rename(columns={'Name':'TagName'})[['Id', 'TagName']], left_on="TagId", right_on="Id", how="left").drop(['Id_x','Id_y', 'Id', 'TagId', 'CompetitionId'], axis=1);
df_data.head()
# ok first I'm interested in looking at popularity of tags (i know the Tags.csv pretty much shows this, but wanted some practice and should check my work against Tags # competitions)

# lets drop tabular data since its not that important and plot occurances of each tag
pd.value_counts(df_data['TagName']).drop('tabular data').plot(kind="bar", figsize=(25, 4))
plt.show()
# now I want to look at total submissions by tag
counts = df_data.groupby('TagName')[['TotalSubmissions']].sum();
# drop 'tabular data' tag here too for consistency
counts.drop('tabular data').sort_values(by="TotalSubmissions", ascending=False).plot(kind="bar", figsize=(25, 4));
plt.show()

counts.sort_values(by="TotalSubmissions", ascending=False).head()
# banking and housing in particular look like popular tags in terms of # submissions...wonder if the reward $ for those tags might explain this?
# lets look into the reward amounts by tag
counts = df_data.groupby('TagName')[['RewardQuantity']].sum(); # we should just filter by USD reward type 
# drop 'tabular data' tag here too for consistency
counts.drop('tabular data').sort_values(by="RewardQuantity", ascending=False).plot(kind="bar", figsize=(25, 4));
plt.show()

counts.sort_values(by="RewardQuantity", ascending=False).head()
# so housing has had some good rewards...banking not so much though...should check my calc for reward amount
# if I had more time, I'd clean these up to be more accurate and dive more into how reward $ correlates to things like # submissions and possibly
# create a training set to predict # of submissions to a competiton based on Tag and reward $