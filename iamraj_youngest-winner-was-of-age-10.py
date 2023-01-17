import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
athlete_df = pd.read_csv('../input/athlete_events.csv')
# Lets look at the athletes data file
athlete_df.head()
# How many total records in the data set
total_records = athlete_df.shape[0]
print("Total Records : {}".format(total_records))
#create a new df group by Year and get the unique number values
season_df = athlete_df.groupby('Year').nunique()
season_df.head(10)
# df to check if there were two games in the same year
two_games_df = season_df[ season_df['Games'] == 2]

# df to check if there were any games where participants were all from the same gender
one_gender_df = season_df[ season_df['Sex'] == 1]
years = two_games_df.index.values
print("The following years had two games in the same year. Winter and Summer games.\n")
for i in years:
    print(i)
years = one_gender_df.index.values
print("The following year(s) had participants only from one gender.\n")
for i in years:
    print(i)
# Lets check the year 1896 to see the gender of the participants of that year
athlete_df[athlete_df['Year'] == 1896].Sex.value_counts()
# df of all medal winners
winners_df = athlete_df[ athlete_df.Medal.notnull() ]

#oldest athlete to win a medal in the games
oldest_medal_winner = winners_df.loc [winners_df ['Age'].idxmax()]
print("The oldest athlete to win a medal : \n")
oldest_medal_winner[['Name','Age','Team','Year','Sport','Event']]
#oldest athlete to participate in the games
oldest_athlete = athlete_df.loc[athlete_df['Age'].idxmax()]
print("The oldest athlete to participate in a game : \n")
oldest_athlete[['Name','Age','Team','Year','Sport','Event']]
# Youngest athlete to participate in the games history
youngest_athlete = athlete_df.loc[athlete_df['Age'].idxmin()]
print("The youngest athlete to participate in a game: \n")
youngest_athlete[['Name','Age','Team','Year','Sport','Event']]
# Youngest Athlete win a medal in the games
youngest_medal_winner = winners_df.loc [winners_df ['Age'].idxmin()]
print("The youngest athlete to win a medal : \n")
youngest_medal_winner[['Name','Age','NOC','Team','Year','Sport','Event','Medal']]
female_winners_df = athlete_df [ (athlete_df['Sex'] == 'F') & (athlete_df['Medal'].notnull())]
youngest_fem_winner = female_winners_df.loc [ female_winners_df ['Age'].idxmin()]

print("Youngest Female Athlete to win a medal in the games : \n")
youngest_fem_winner[['Name','Age','NOC','Team','Year','Sport','Event','Medal']]
# lets look at the games from 2000 onwards
this_century_df = athlete_df[ athlete_df['Year'] > 1999]
# how many athletes participated over the years
sns.countplot(x='Year', data=this_century_df, order = this_century_df['Year'].value_counts(ascending=True).index);
# Medals won my Male and Female athletes for the games from Year 2000
sns.catplot(y="Sex",hue="Medal", data=this_century_df, kind="count", palette="copper_r");