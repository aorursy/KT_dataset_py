# importing packages

import pandas as pd

import numpy as np

from scipy.interpolate import spline



# for plots

import matplotlib.pyplot as plt

from matplotlib.dates import date2num



# for date and time processing

import datetime



# for statistical graphs

import seaborn as sns



# for machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# getting the dataset 

train = pd.read_csv("../input/FMEL_Dataset.csv")
# checking the dataset

print (train.head (10))
# checking the dataset

print (train.tail (10))

# we have 36305 rows (starts with 0)
# checking infos about data 

print (train.info())

# from the output of this line, we can learn that there is no null values in the dataset
# for statistical descriptions 

print (train.describe())
# for details of non-numeric attributes

print (train.describe(include=['O']))

# from the output of this line, we can learn that most number of matches were played in 2010-11 

# season, "Sporting de Gijon" is the team which played most number of the matches in both 

# local and outside, on 02/06/1991, most number of matches were played on a single day
# creating new feature "local_won"

# I am creating a new column called "local_won" based on "localGoals" and "visitorGoals" columns,

# which will tell us if the match is won by local team or not. If that particular match is won by 

# local team, 1 is shown if not 0.



def f(row):

    if row['localGoals'] > row['visitorGoals']:

        val = 1

    else:

        val = 0

    return val

train['local_won'] = train.apply(f, axis=1)

# check the "local_won" column added at the last

print (train.head ())
# creating new feature "visitor_won"

# I am creating a new column called "visitor_won" based on "localGoals" and "visitorGoals" columns,

# which will tell us if the match is won by visitor team or not. If that particular match is won by 

# visitor team, 1 is shown if not 0.



def g(row):

    if row['visitorGoals'] > row['localGoals']:

        val = 1

    else:

        val = 0

    return val

train ["visitor_won"] = train.apply(g, axis=1)

# check the "visitor_won" column added at the last

print (train.head ())
# creating new feature "match_draw"

# I am creating a new column called "match_draw" based on "localGoals" and "visitorGoals" columns,

# which will tell us if the match was tie or not. If that particular match was not won by anyone, 

# 1 is shown if not 0.



def h(row):

    if row['visitorGoals'] == row['localGoals']:

        val = 1

    else:

        val = 0

    return val

train ["match_draw"] = train.apply(h, axis=1)
# check the "match_draw" column added at the last

print (train.head ())
# creating new feature "total_goals"

# since it is the direct addition of two columns, we are not using IF statement

train['total_goals'] = train['visitorGoals'] + train['localGoals']
# check newly created "total_goals" column

train.head ()
# checking how the total goals tread has been changed over seasons

fig = plt.figure(figsize=(20, 10))

batman = train.groupby('season')['total_goals'].sum()

batman.plot (kind="line", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.show ()

# from the below chart, we can learn that total number of goals scored in every season have gone up 

# over the past 47 years
# lets check the trend of total matches played in these 47 years in each season

fig = plt.figure(figsize=(20, 10))

superman = train.groupby('season')['division'].count()

superman.plot (kind="line", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.show ()

# from the below chart, we can confirm that total number of matches played in each season also

# went up. This is one of the main reason for increase in total number of goals 

# checking which team won mostly

# by analysing the output, we can confirm that local team is most likely to win the match

winning_percent = train[['season', 'local_won', 'visitor_won', 'match_draw']].groupby(['season'], 

     as_index=False).mean().sort_values(by='season')

print (winning_percent)
# showing the average of local_won, visitor_won and match_draw

# this clearly shows that local team have won 51.1% of the overall matches played.

# 21.3% matches were won by visitor teams and 27.5% matches were went tie

print (winning_percent[['local_won', 'visitor_won', 'match_draw']].mean())
# chart to display teams that won most of the time

objects = ('local_won', 'visitor_won', 'match_draw')

y_pos = np.arange(len(objects))

performance = [0.511558,0.213002,0.275440]

 

plt.bar(y_pos, performance, align='center', alpha=0.7)

plt.xticks(y_pos, objects)

plt.ylabel('Percentage')

plt.title('Team Winning Percentage')

plt.show ()



# its obvious from the below chart that teams that played in their own land are twice more

# likely to win the match
# plotting the bar graph to prove this

pos = list(range(len(winning_percent['local_won'])))

width = 0.25



fig, ax = plt.subplots(figsize=(20,5))



plt.bar(pos, winning_percent['local_won'], width, alpha=0.75, color='#006400')

plt.bar([p + width for p in pos], winning_percent['visitor_won'], width, alpha=0.75, color='#000080')

plt.bar([p + width*2 for p in pos], winning_percent['match_draw'], width, alpha=0.75, color='#8B0000')



ax.set_title('OVERALL % OF MATCHES WON BY TEAMS EVERY YEAR')

ax.set_ylabel('% of total matches won')



ax.set_xticks([p + 1 * width for p in pos])

ax.set_xticklabels(winning_percent['season'], rotation=90, fontsize = 20)



plt.legend(['local_won', 'visitor_won', 'match_draw'], loc='upper right')



plt.grid()

plt.show()



# from the below chart, its very obvious that when teams get to play in their own land, 

# tends to win the match than a team that goes to a foreign land for a match.
# lets check which team won most of the matches when played in home land

teams = train.groupby('localTeam')['local_won'].sum().sort_values(ascending=True)

print (teams)

# below are the teams won most of the times in their homeland:

# lets plot this data in a horizontal chart

fig = plt.figure(figsize=(20,50))

teams = train.groupby('localTeam')['local_won'].sum().sort_values(ascending=True)

teams.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.show ()

# from the below chart, its clear that Real Madrid is the team that won most of the matches, followed 

# by Barcelona very closely
# lets check which team won most of the matches when played in home land

teams1 = train.groupby('visitorTeam')['visitor_won'].sum()

print (teams)

# below are the teams won most of the times in their homeland:
# lets plot this data in a horizontal chart

fig = plt.figure(figsize=(20,50))

teams1 = train.groupby('visitorTeam')['visitor_won'].sum().sort_values(ascending=True)

teams1.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.show ()

# from the below chart, its clear that Real Madrid is the team that won most of the matches, followed 

# by Barcelona very closely in foreign land too
# I am going to analyze how many matches are totally played in every month

train['game_month'] = pd.to_datetime(train['date'], format='%d/%m/%Y')

train['game_date'] = train['game_month'].dt.month

train['game_weekday'] = train['game_month'].dt.weekday

train.groupby([train['game_month'].dt.month])['round'].count().plot(kind='bar')
# from the above chart, we can conclude that very less number of matches were played in 

# June and July months. They are schools, colleges reopening months after summer holidays.

# That might be a reason.
# Looks like most games are played on the weekend :)

train.groupby('game_weekday')['round'].count().plot(kind='bar')
# Most of the matches were played on weekends.