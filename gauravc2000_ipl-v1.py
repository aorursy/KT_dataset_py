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
import pandas as pd
Ball_by_Ball = pd.read_csv("../input/indian-premier-league-csv-dataset/Ball_by_Ball.csv")
Match = pd.read_csv("../input/indian-premier-league-csv-dataset/Match.csv")
Player = pd.read_csv("../input/indian-premier-league-csv-dataset/Player.csv")
Player_Match = pd.read_csv("../input/indian-premier-league-csv-dataset/Player_Match.csv")
Season = pd.read_csv("../input/indian-premier-league-csv-dataset/Season.csv")
Team = pd.read_csv("../input/indian-premier-league-csv-dataset/Team.csv")


df1 = Ball_by_Ball.groupby('Match_Id').Innings_Id.unique()
df1
df1 = Ball_by_Ball.groupby(['Match_Id', 'Innings_Id']).Team_Batting_Id.nunique()
df1[df1 !=1].head()
Ball_by_Ball.head()
## Rows and columns 
print("\tNumber of rows", Ball_by_Ball.shape[0])
print("\tNumber of columns", Ball_by_Ball.shape[1])
## See the first few rows 
Ball_by_Ball.head()
## Unique matches
print("\tNumber of unique matches", Ball_by_Ball.Match_Id.nunique())
## Number of matches Matches by season 

Match.groupby('Season_Id').Match_Id.count()
## Which seasons? 
Season.Season_Year.unique()
## Innings should be either 1 or 2 for T20
print("\tUnique innings", Ball_by_Ball.Innings_Id.unique())
# What is innings 3,4? 
print ("\t Number of rows by innings id")
print (Ball_by_Ball.Innings_Id.value_counts())

## How many matches have innings 3,4?
print ("\tMatches with 3rd or 4th innings", Ball_by_Ball.loc[Ball_by_Ball.Innings_Id.isin([3,4])].Match_Id.unique())