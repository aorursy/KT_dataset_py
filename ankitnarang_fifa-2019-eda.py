# Import necessary libraries

import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.set(style='ticks')



# Customize colors

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"] # defining the colour palette

flatui = sns.color_palette(flatui)
from wordcloud import WordCloud  
# Read the dataset

df = pd.read_csv("../input/fifa-2019/FIFA_data.csv")

df.head()
df.info()
print(df.columns)
df.shape
def print_full(x):

    pd.set_option('display.max_rows', len(x))

    print(x)

    pd.reset_option('display.max_rows')
all = df.isnull().sum()

print_full(all)
import missingno as msno

msno.matrix(df)
#msno.heatmap(df.sample(1000))
# Plot the heatmap of the correlation between columns

fig, ax = plt.subplots(figsize=(25,25))

sns.heatmap(df.corr(), annot = True, linewidths=0.5, linecolor="green", fmt = '.1f', ax=ax)

plt.show()
# plot the word cloud for nationality of players

plt.subplots(figsize=(25,15))



wordcloud = WordCloud(

                        background_color="white",

                        width = 1920,

                        height = 1080,

                        ).generate(" ".join(df.Nationality))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
# impute missing values

df['Club'].fillna('No Club', inplace=True)

df['Position'].fillna('ST', inplace=True)
# Impute missing values by mean

missing_by_mean = df.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',

                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                                 'GKKicking', 'GKPositioning', 'GKReflexes']]

missing_by_mean[:5]
for i in missing_by_mean.columns:

    df[i].fillna(df[i].mean(), inplace=True)
# Impute missing values in categorical variables by mode

missing_by_mode = df.loc[: , ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]

for i in missing_by_mode.columns:

    df[i].fillna(df[i].mode()[0], inplace=True)
# impute other variable, discrete and continuous with median

impute_by_median = df.loc[: , ['Weak Foot', 'Skill Moves']]



for i in impute_by_median.columns:

    df[i].fillna(df[i].median(), inplace=True)
df.columns[df.isnull().any()]
df.fillna(0, inplace = True) # Filling the remaining  missing values with zero

df.isnull().all().sum()
# functions to get the rounded values from different columns

def defending(data):

    return int(round((data[['Marking', 'StandingTackle','SlidingTackle']].mean()).mean()))



def general(data):

    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 

                               'BallControl']].mean()).mean()))



def mental(data):

    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 

                               'Vision','Composure']].mean()).mean()))



def passing(data):

    return int(round((data[['Crossing', 'ShortPassing', 

                               'LongPassing']].mean()).mean()))



def mobility(data):

    return int(round((data[['Acceleration', 'SprintSpeed', 

                               'Agility','Reactions']].mean()).mean()))

def power(data):

    return int(round((data[['Balance', 'Jumping', 'Stamina', 

                               'Strength']].mean()).mean()))



def rating(data):

    return int(round((data[['Potential', 'Overall']].mean()).mean()))



def shooting(data):

    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 

                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))
# Rename columns

df.rename(columns = {'Club Logo': 'Club_Logo'}, inplace=True)
# adding new columns to the data



df['Defending'] = df.apply(defending, axis=1)

df['General'] = df.apply(general, axis=1)

df['Mental'] = df.apply(mental, axis=1)

df['Passing'] = df.apply(passing, axis=1)

df['Mobility'] = df.apply(mobility, axis=1)

df['Power'] = df.apply(power, axis=1)

df['Rating'] = df.apply(rating, axis=1)

df['Shooting'] = df.apply(shooting, axis=1)

# creating the players dataset

players = df[['Name','Defending','General','Mental','Passing',

                'Mobility','Power','Rating','Shooting','Flag','Age',

                'Nationality', 'Photo', 'Club_Logo', 'Club']]



players.head(20)
# Different position taken be players



plt.figure(figsize=(18,8))

plt.style.use("fivethirtyeight")

ax = sns.countplot('Position', data=df, palette='dark')

ax.set_xlabel("Different positions in football")

ax.set_ylabel("Count of players")

ax.set_title("Comparision of positions and players")

plt.show()
# Plot count of players based on their heights



plt.figure(figsize=(18,8))

ax = sns.countplot('Height', data=df, palette='bone')

ax.set_xlabel("Height of players")

ax.set_ylabel("Count of players")

ax.set_title("Number of players and their height")

plt.show()
# Plot count of players based on their work rates



plt.figure(figsize=(18,8))

ax = sns.countplot('Work Rate', data=df, palette='bone')

ax.set_xlabel("Different work rates of the players")

ax.set_ylabel("Count of players")

ax.set_title("Work rate of the players")

plt.show()
Total_players = df.groupby("Nationality")["ID"].count().sort_values(ascending=False).head(10)

Total_players = pd.DataFrame(Total_players)

Total_players.rename(columns = {"ID":"Number"}, inplace=True)

Total_players
# Plot count of players based on their work rates



plt.figure(figsize=(18,8))

ax = sns.barplot(x = Total_players.index, y="Number", data=Total_players, palette='bone')

ax.set_xlabel("Nationality")

ax.set_ylabel("Count of players")

ax.set_title("Number of players")

plt.show()
Top_ratings = df.groupby(['Club'])["Rating"].mean().sort_values(ascending=False)[:10]

Top_ratings = pd.DataFrame(Top_ratings)

Top_ratings
# Plot group with top ratings



plt.figure(figsize=(18,8))

ax = sns.lineplot(x = Top_ratings.index, y="Rating", data=Top_ratings, palette='bone')

ax.set_xlabel("Groups")

ax.set_ylabel("Ratings")

ax.set_title("Ratings by groups")

plt.show()
print(df.columns)

df.head()
penalty = df.groupby("Name")["Penalties"].count().sort_values(ascending=False)[:10]

penalty = pd.DataFrame(penalty)

penalty
# Plot the number of penalties



plt.figure(figsize=(18,8))

ax = sns.barplot(x = penalty.index, y="Penalties", data=penalty, palette='bone')

ax.set_xlabel("Name")

ax.set_ylabel("Penalties")

ax.set_title("Top penalties")

plt.show()
penalty
# showing the name of the players which occurs the most number of times from the first 20 names

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Name[0:20]))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('players.png')

plt.show()
# checking which clubs have been mentioned the most

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Club))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('players.png')

plt.show()