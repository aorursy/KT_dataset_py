# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', 500)
import seaborn as sns

import matplotlib.pyplot as plt
from kaggle.competitions import nflrush



# You can only call make_env() once, so don't lose it!

env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train_df.shape
train_df.head()
# train_df[train_df['PlayId']==20181007011551]
train_df["GameId"].value_counts()
print("From ",train_df["GameId"].astype(str).str[:6].min(),"To ",train_df["GameId"].astype(str).str[:6].max())
pd.Series(train_df["GameId"].unique()).astype(str).str[:6].value_counts().sort_index().plot(kind="barh")
print(len(train_df["PlayId"].unique()))

print(int(509762/22))
team_df=train_df[['PlayId','Team','NflId','Yards']].merge(

    train_df[['PlayId','NflIdRusher']].drop_duplicates(),left_on=['PlayId','NflId'],

    right_on=['PlayId','NflIdRusher'],how='inner')
team_df['Team'].value_counts().plot(kind="barh",title='Offense times', figsize=(10,2))
print('Average yards gained by home team : ',team_df[team_df["Team"]=='home']['Yards'].mean())

print('Average yards gained by away team : ',team_df[team_df["Team"]=='away']['Yards'].mean())

print('Median yards gained by home team :  ',team_df[team_df["Team"]=='home']['Yards'].median())

print('Median yards gained by away team :  ',team_df[team_df["Team"]=='away']['Yards'].median())
f, axes = plt.subplots(1, 2, figsize=(10,4))

sns.distplot(team_df[team_df["Team"]=='home']['Yards'],ax=axes[0],kde=None).set_title("Home")

sns.distplot(team_df[team_df["Team"]=='away']['Yards'],ax=axes[1],kde=None).set_title("Away")

print("Gained Yards Distplot:")
train_df[['X', 'Y', 'S', 'A', 'Dis', 'Orientation','Dir']].describe()
print("Orientation: ",train_df[train_df["Orientation"].isnull()]["DisplayName"].drop_duplicates().values)



print("Dir: ",train_df[train_df["Dir"].isnull()]["DisplayName"].drop_duplicates().values)
f, axes = plt.subplots(2, 4, figsize=(18,9))

sns.distplot(train_df["X"],ax=axes[0,0])

sns.distplot(train_df["Y"],ax=axes[0,1])

sns.distplot(train_df["S"],ax=axes[0,2])

sns.distplot(train_df["A"],ax=axes[0,3])

sns.distplot(train_df["Dis"],ax=axes[1,0])

sns.distplot(train_df[~train_df["Orientation"].isnull()]["Orientation"],ax=axes[1,1])

sns.distplot(train_df[~train_df["Dir"].isnull()]["Dir"],ax=axes[1,2])
train_df[train_df["DisplayName"]=="Michael Thomas"][["DisplayName","NflId"]].drop_duplicates()
sns.distplot(train_df["JerseyNumber"],kde=None)
train_df["Season"].value_counts().plot(kind="barh")
sns.distplot(train_df["YardLine"],kde=None)
print("The peak: ",train_df["YardLine"].value_counts().max())
train_df.groupby("PlayId")[["YardLine","Yards"]].max().corr()
train_df["Quarter"].value_counts().plot(kind="barh")
train_df.groupby("PlayId")[["Quarter","Yards"]].max().corr()
train_df["GameClock"].value_counts()[:5]
train_df["PossessionTeam"].value_counts().plot(kind="bar",figsize=(20,5))
train_df[["PlayId","PossessionTeam","Yards"]].drop_duplicates().groupby(["PossessionTeam"])["Yards"].mean().sort_values().plot(kind="bar",figsize=(20,5))
train_df[["PlayId","PossessionTeam","Yards"]].drop_duplicates().groupby(["PossessionTeam"])["Yards"].median().sort_values().plot(kind="bar",figsize=(20,5))
train_df["Down"].value_counts().plot(kind="barh")
train_df[["PlayId","Down","Yards"]].drop_duplicates().groupby(["Down"])["Yards"].mean().sort_values().plot(kind="barh")
train_df["Distance"].value_counts().sort_index().plot(kind="bar",figsize=(20,5))
train_df.groupby("PlayId")[["Distance","Yards"]].max().corr()
print("Distance > 10 : ",sum(train_df["Distance"]>10)/len(train_df))
train_df["FieldPosition"].value_counts().plot(kind="bar",figsize=(20,5))
f, axes = plt.subplots(1, 2, figsize=(10,4))

sns.distplot(train_df["HomeScoreBeforePlay"],ax=axes[0],kde=None)

sns.distplot(train_df["VisitorScoreBeforePlay"],ax=axes[1],kde=None)
print(len(train_df["NflIdRusher"].value_counts()),"unique players")
train_df["OffenseFormation"].value_counts().plot(kind="barh")
train_df["OffensePersonnel"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["DefendersInTheBox"].value_counts().sort_index(ascending=False).plot(kind="barh")
train_df["DefensePersonnel"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["PlayDirection"].value_counts().plot(kind="barh")
train_df[["TimeHandoff","TimeSnap"]][:5]
sns.distplot(train_df["Yards"],kde=None)
f, axes = plt.subplots(1, 2, figsize=(10,4))

sns.distplot((train_df["PlayerHeight"].str[:1].astype(int)*12+train_df["PlayerHeight"].str[-1:].astype(int)),kde=None,ax=axes[0])

sns.distplot(train_df["PlayerWeight"],kde=None,ax=axes[1])
train_df["PlayerCollegeName"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["Position"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["HomeTeamAbbr"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["VisitorTeamAbbr"].value_counts().plot(kind="bar",figsize=(20,5))
sns.distplot(train_df["Week"],kde=None)
train_df["Stadium"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["Location"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["StadiumType"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["Turf"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["GameWeather"].value_counts().plot(kind="bar",figsize=(20,5))
f, axes = plt.subplots(1, 2, figsize=(10,4))

sns.distplot(train_df["Temperature"].fillna(0),kde=None,ax=axes[0])

sns.distplot(train_df["Humidity"].fillna(0),kde=None,ax=axes[1])
train_df["WindSpeed"].value_counts().plot(kind="bar",figsize=(20,5))
train_df["WindDirection"].value_counts().plot(kind="bar",figsize=(20,5))