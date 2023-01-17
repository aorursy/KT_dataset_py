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
import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns

# from kaggle.competitions import nflrush



# # You can only call make_env() once, so don't lose it!

# env = nflrush.make_env()



pd.set_option('max_columns', 100)

data_path = "/kaggle/input/nfl-big-data-bowl-2020/"



train = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv", low_memory=False)

train.head()
# How many season, games, plays and weeks does the data accumulate ?

print("Seasons: {0}, Games: {1}, Plays: {2}, Weeks: {3}".format(train.Season.nunique(), 

                                                                train.GameId.nunique(), 

                                                                train.PlayId.nunique(),

                                                                train.Week.nunique()))
# How many plays does a game have? -- It varies across each game, see the cell below for more details

print("# Plays in a game: ", train[train['GameId']==2017090700].PlayId.nunique())



# How many players data does a play contain?

print("# Players in a play", train[train['PlayId']==20170907000118].NflId.nunique())
# Does the number of plays vary across games? -- No

train.groupby(['GameId'], as_index=False).agg({'PlayId': 'nunique'}).head()
# Does team vary across each play? -- Yes, it denotes the team of the nfl player

train[train['PlayId']==20170907000118].Team.value_counts()
# How are the jersey numbers distributed across players?

train[train['PlayId']==20170907000118].JerseyNumber.values.ravel()
# How are the X, Y position distributed and their min and max values in a game



print(train[train['PlayId']==20170907000118].X.values.ravel())

print(train[train['PlayId']==20170907000118].Y.values.ravel())



print("X min: ", train[train['GameId']==2017090700].X.min())

print("X max: ", train[train['GameId']==2017090700].X.max())



print("Y min: ", train[train['GameId']==2017090700].Y.min())

print("Y max: ", train[train['GameId']==2017090700].Y.max())
# Get a glimpse of speed and acceleration values in a play? Are they different?

print("Speed: ", train[train['PlayId']==20170907000118].S.values.ravel())

print("Acceleration: ", train[train['PlayId']==20170907000118].A.values.ravel())

# Get a glimpse of distance, orientation and direction values in a play? Are the different?

print("Distance: ", train[train['PlayId']==20170907000118].Dis.values.ravel())

print("Orientation: ", train[train['PlayId']==20170907000118].Orientation.values.ravel())

print("Direction: ", train[train['PlayId']==20170907000118].Dir.values.ravel())
# Distribution of data and the plays in a season

print(train['Season'].value_counts())

print("\n Number of games in a season: ")

print(train.groupby(['Season'], as_index=False)['GameId'].nunique())

print("\n Number of plays in a season: ")

print(train['Season'].value_counts()/22)
## YardLine (line of scrimmage) is unique for each Play

train[train['GameId']==2017090700].YardLine.value_counts().head()
train.Quarter.value_counts()

## Why is there more number of 4th, 3rd quarters than 2nd quarter?
## GameClock, TimeHandOff, TimeSnap are all unique at PlayId level

print("Min gameclock time: ", train[train['GameId']==2017090700].GameClock.min())

print("Max gameclock time: ", train[train['GameId']==2017090700].GameClock.max())

train[train['GameId']==2017090700].GameClock.value_counts().head()
# Number of plays a team is in possession of the ball

train[train['GameId']==2017090700].PossessionTeam.value_counts()/22
# Number of plays does a down contain

train[train['GameId']==2017090700].Down.value_counts()/22

# Most of the plays are on 1st and 2nd down
# What are the top 5 distance required for the 1st down

(train[train['GameId']==2017090700].Distance.value_counts()/22).head()
# Are field positions similar to the Possession team?

train[train['GameId']==2017090700].FieldPosition.value_counts(dropna=False)/22
# How are the Home score and visitor score before play distributed?

print(train[train['GameId']==2017090700].HomeScoreBeforePlay.value_counts()/22)

print(train[train['GameId']==2017090700].VisitorScoreBeforePlay.value_counts()/22)
# Are there some specialized rushers in the game(teams)

train[train['GameId']==2017090700].NflIdRusher.value_counts()/22
# Types of offense formations? Are there any special offense formations or only general?

train[train['GameId']==2017090700].OffenseFormation.value_counts()/22
# What different positions are the offense personnel located in each game/play

train[train['GameId']==2017090700].OffensePersonnel.value_counts()/22
# Does the defenders in the box vary with the play in a game?

train[train['GameId']==2017090700].DefendersInTheBox.value_counts()/22
# What different positions are the defense personnel located in each game/play

train[train['GameId']==2017090700].DefensePersonnel.value_counts()/22
# Is play direction comparative to the possession team's (32,20)? - No

train[train['GameId']==2017090700].PlayDirection.value_counts()/22

# No. Do the teams change their sides for each quarter?
# Frequency of gaining / loosing yards in a game

train[train['GameId']==2017090700].Yards.value_counts()/22
# Does player vary from play to play in a game? - yes

# How does their height distribution for a single game?

train[train['GameId']==2017090700].PlayerHeight.value_counts()
# More distinct player weights compared to player height

len(train[train['GameId']==2017090700].PlayerWeight.value_counts())
# What is the birthdate format? we can subtract this with the gameId date field to get age

train[train['PlayId']==20170907000118].PlayerBirthDate.head()
# Understand the player college name feature

(train[train['GameId']==2017090700].PlayerCollegeName.value_counts()/22).head()
# Position for a play

train[train['PlayId']==20170907000118].Position.value_counts()
# List of all the positions in the train data set

train.Position.value_counts().index.values
# Number of home teams and visitor teams

train[train['Season']==2018].HomeTeamAbbr.nunique(), \

train[train['Season']==2018].VisitorTeamAbbr.nunique()

# Both are equal - games are played in all the team's home ground
# Number of weeks the nfl game is played

train.Week.unique()
# what are the 43 stadiums in which season 2018 is played?

train[train['Season']==2018].Stadium.unique()
# What are the (53) locations at which season 2018 was played?

train[train['Season']==2018].Location.unique()

# -- There are duplicates in these 53 locations
# Which stadium type is more played on? Are there any duplicates?

# train[train['Season']==2018].StadiumType.value_counts()/22

train[train['Season']==2018].StadiumType.unique()

# On string formatting and some grouping the number of stadium types can be reduced
# What are the different turfs on which the games are played?

train[train['Season']==2018].Turf.value_counts()

# There are some duplicate categories which can be grouped
# What are the different game weather and how are they distributed?

train[train['Season']==2018].GameWeather.value_counts()/22
# Understanding temperature

train[train['Season']==2018].Temperature.nunique(), train[train['Season']==2018].Temperature.min(), \

train[train['Season']==2018].Temperature.max()
train[train['Season']==2018].Humidity.nunique(), train[train['Season']==2018].Humidity.min(),\

train[train['Season']==2018].Humidity.max()
train[train['Season']==2018].WindSpeed.value_counts()/22
train[train['Season']==2018].WindDirection.value_counts()
# https://en.wikipedia.org/wiki/Uniform_number_(American_football)

# Jersey numbers are provided based on the player roles (also, eligibility of pass/receive ball)

# For offensive team,

#### Jersey numbers 50-79 & 90-99 are ineligible pass receivers

#### Jersey numbers 1-49 & 80-89 are eligible pass receivers

train[train['GameId']==2017090700].JerseyNumber.nunique()
# ## Generalising the play direction to unidirectional offense

# ## Identify if the player carries the ball

# train['ToLeft'] = (train['PlayDirection'] == 'left').astype('uint8')

# train['IsBallCarrier'] = (train['NflId'] == train['NflIdRusher']).astype('uint8')
# ## Updating the team abbreviation

# old_abbr = ["ARI", "BAL", "CLE", "HOU"]

# new_abbr = ["ARZ", "BLT", "CLV", "HST"]

# for aa, bb in zip(old_abbr, new_abbr):

#     print((train['VisitorTeamAbbr'] == aa).sum(), (train['HomeTeamAbbr'] == aa).sum())

#     train.loc[train['VisitorTeamAbbr'] == aa,'VisitorTeamAbbr'] = bb

#     train.loc[train['HomeTeamAbbr'] == aa, 'HomeTeamAbbr'] = bb
# ## Creating a variable to store the team that is on offense

# train_1 = train.copy()

# def to_ofn(a):

#     if a[0] == a[1]:

#         return "home"

#     else:

#         return "away"

    

# train_1['TeamOnOffense'] = train_1[['PossessionTeam','HomeTeamAbbr']].apply(to_ofn, axis=1)

# # train_1.head()
# ## Is the player on offense

# train_1['IsOnOffense'] = (train_1['Team'] == train_1['TeamOnOffense']).astype('uint8')



# ## Number of yards from their own team goal

# def yd_own_goal(x):

#     if x[0] == x[1]:

#         return x[2]

#     else:

#         return 50 + (50 - x[2])

# train_1['YardsFromOwnGoal'] = train_1[['FieldPosition','PossessionTeam','YardLine']].apply(yd_own_goal, 

#                                                                                            axis=1)

# # train_1.head()
# ## Adjusting for 50 yards yardline

# def check_50yd(y):

#     if y[1] == 50:

#         return 50

#     else:

#         return y[0]

# train_1['YardsFromOwnGoal'] = train_1[['YardsFromOwnGoal', 'YardLine']].apply(check_50yd, axis=1)



# # train_1.head()
# ## Standardize the X and Y values

# def stdize_x(a):

#     if a[0] == 1:

#         return 120 - a[1] - 10

#     else:

#         return a[1] - 10



# def stdize_y(b):

#     if b[0] == 1:

#         return (160/3) - b[1]

#     else:

#         return b[1]



# train_1['X_std'] = train_1[['ToLeft', 'X']].apply(stdize_x, axis=1)

# train_1['Y_std'] = train_1[['ToLeft', 'Y']].apply(stdize_y, axis=1)



# train_1.head()