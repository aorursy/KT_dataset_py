#import required packages

%matplotlib inline

import matplotlib.pyplot as plt 

import plotly

import numpy as np

import plotly.graph_objs as go

import seaborn as sns

import pandas as pd

import datetime as dt

import warnings

warnings.filterwarnings('ignore')



#read the data file

df1 = pd.read_csv('../input/metacritic_games.csv', delimiter=',')



#add total critics and total users

df1['total_critics'] = df1['positive_critics'] + df1['neutral_critics'] + df1['negative_critics'] 

df1['total_users'] = df1['positive_users'] + df1['neutral_users'] + df1['negative_users']



#make release date a datetime

df1['release_date'] = pd.to_datetime(df1['release_date'])



#quick checkup that the data looks good

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns. Below is a sample of 5 random rows.')

df1.sample(5)
#SUB-PROJECT 1: create a visualization showing the total percentage of positive/neutral/negative critic scores and user scores



#get the percentage of each review type

percent_positive_reviews = [df1['positive_critics'].sum()/df1['total_critics'].sum()*100, df1['positive_users'].sum()/df1['total_users'].sum()*100]

percent_neutral_reviews = [df1['neutral_critics'].sum()/df1['total_critics'].sum()*100, df1['neutral_users'].sum()/df1['total_users'].sum()*100]

percent_negative_reviews = [df1['negative_critics'].sum()/df1['total_critics'].sum()*100, df1['negative_users'].sum()/df1['total_users'].sum()*100]



#display data

x_shape = [0,1]

x_names = ('Critics','Users')

plt.bar(x_shape, percent_positive_reviews, color='g', edgecolor='white', label = 'Postive reviews')

plt.bar(x_shape, percent_neutral_reviews, bottom=percent_positive_reviews, color='y', edgecolor='white', label = 'Neutral reviews')

plt.bar(x_shape, percent_negative_reviews, bottom=np.add(percent_positive_reviews, percent_neutral_reviews), color='r', edgecolor='white', label = 'Negative reviews')

plt.xticks(x_shape, x_names)

plt.legend(loc='right', bbox_to_anchor=(1.45, 0.5))

plt.show()
#SUB-PROJECT 2: create a visualization showing number of games in each genre and platform

#Also decide if this looks better with the filter at over 300 or over 100 for the "Other" category in the Genre Breakdown



#count occurences of each genre and platform

genre_series = df1['genre'].value_counts().sort_values(ascending=True)

platform_series = df1['platform'].value_counts().sort_values(ascending=True)



#filter out genres with less than 300 occurences and group them in an "Other category" ("Misc" has 282)

large_genres = genre_series[genre_series > 300] 

other_count = 0

for x in genre_series:

    if x < 300:

        other_count = other_count + x

large_genres.set_value('Other', other_count)



#display data

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.pie(large_genres.values, labels=large_genres.index, autopct='%1.0f%%')

ax1.set_title("Genre Breakdown")

ax2.pie(platform_series.values, labels=platform_series.index, autopct='%1.0f%%')

ax2.set_title("Platform Breakdown")

plt.show()
#SUB-PROJECT 3: create a visualization of the correlation between total critic reviews vs metascore



#display data

df1.plot(kind='scatter', x='total_critics', y='metascore')

plt.show()
#SUB-PROJECT 4: create a visualization of game score vs. "popularity", i.e., user reviews. Need to study additional data to determine how sales/active players correlate to user reviews



#manually marking notable games on the graph to appear red

notable_points_df = pd.DataFrame(df1[(df1['game'] == 'Diablo III') | (df1['game'] == 'Infestation: Survivor Stories (The War Z)') | (df1['game'] == 'Star Wars Battlefront II')])



#display data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

df1.plot(kind='scatter', x='total_users', y='user_score', ax = ax1)

notable_points_df.plot(kind='scatter', x='total_users', y='user_score', ax = ax1, color = 'r')

df1.plot(kind='scatter', x='total_users', y='metascore', ax = ax2)

notable_points_df.plot(kind='scatter', x='total_users', y='metascore', ax = ax2, color = 'r')



plt.show()
#SUB-PROJECT 5: visualization with rating on the x-axis, and metascore + user score on the y-axis. 

#perhaps I can figure out how to put these both on the same graph maybe with a violinplot where each "violin" is split down the middle?

#Also having trouble putting these side by side. the method I used above doesn't seem to work with seaborn?

#investigate whether difference in metascore for M and T games is statistically significant



#remove RP and AO due to low sample sizes and investigate correlation between genre and score UNFINISHED

#clean_ratings = df1[(df1['rating'] == 'E10+') | (df1['rating'] == 'M') | (df1['rating'] == 'T') | (df1['rating'] == 'E')]



#display data

sns.catplot(x='rating',y='metascore', data = df1, kind="boxen")

sns.catplot(x='rating',y='user_score', data = df1, kind="boxen")

plt.show()
#SUB-PROJECT 6: activity on metacritic by year



#display data

df1['release_date'].groupby(df1['release_date'].dt.year).hist()

plt.show()
#SUB-PROJECT 7. market share of each platform by year. VERY UNFINISHED

#alright wait, why is there a missing slot in 2014. huh. I don't think I understand how map lambda works...

df1['release_date'].map(lambda d: d.year).plot(kind='hist')

plt.show()
#SUB-PROJECT 8. games that critics hated while users loved, and vice versa



#Went back to check the average metascore and user score due to such a large discrepancy in "games users hate" vs. "games users love" at the +/- 25 rating threshold

print("Correlation between user score and metascore: " + str(df1['metascore'].corr(df1['user_score']))[:4])

df1['metascore'].plot(kind = 'kde', xlim = (0, 100), label = ('metascore (mean = ' + (str(df1['metascore'].mean())[:4])) + ')')

df1['user_score'].plot(kind = 'kde', xlim = (0, 100), label = ('user score (mean = ' + (str(df1['user_score'].mean())[:4])) + ')')

plt.xlabel('score')

plt.legend()



#Due to aforementioned discrepancy, we use a threshold of +50 for games users hate

users_hate_df = pd.DataFrame(df1[(df1['metascore'] - df1['user_score'] > 50) & (df1['total_users'] > 5)]) 

users_love_df = pd.DataFrame(df1[(df1['metascore'] - df1['user_score'] < -25) & (df1['total_users'] > 5)])



#display data

plt.show()

print("Games with much higher critic reviews")

display(users_hate_df[['game','developer','platform','genre','release_date','total_critics','total_users','metascore', 'user_score']])

print("Games with much higher user reviews")

display(users_love_df[['game','developer','platform','genre','release_date','total_critics','total_users','metascore', 'user_score']])
#SUB-PROJECT 9. correlation between user score and metascore across popularity buckets, genres, dates. VERY UNFINISHED

print("Correlation between user score and metascore for:")

print("Overall: " + str(df1['metascore'].corr(df1['user_score']))[:4])



#alright, replace this trash-tier code with a loop when you get the chance. should be pretty easy. Then get the correlations for genres, ratings, dates, etc.

print("PC games: " + str(df1[(df1['platform'] == 'PC')].metascore.corr(df1.user_score))[:4])

print("PS4 games: " + str(df1[(df1['platform'] == 'PS4')].metascore.corr(df1.user_score))[:4])

print("XONE games: " + str(df1[(df1['platform'] == 'XONE')].metascore.corr(df1.user_score))[:4])

print("Switch games: " + str(df1[(df1['platform'] == 'Switch')].metascore.corr(df1.user_score))[:4])
#SUB-PROJECT 10 Analyze polarizing games (lots of positive and negative reviews) vs non-polarizing games (lots of middling reviews)



#metacritic considers a score "neutral" if it is between 50 and 75. I manually toyed with numbers to achieve a reasonably-sized result here

polarizing_critics_df = pd.DataFrame(df1[(df1['neutral_critics'] / df1['total_critics'] < .2) & (df1['metascore'] > 50) & (df1['metascore'] < 75) & (df1['total_users'] > 5)]) 

polarizing_users_df = pd.DataFrame(df1[(df1['neutral_users'] == 0) & (df1['user_score'] > 50) & (df1['user_score'] < 75) & (df1['total_users'] > 13)]) 



#display data

print("Games with polarized critic reviews")

display(polarizing_critics_df[['game','developer','platform','genre','release_date','total_critics','total_users', 'metascore', 'user_score']])

print("Games with polarized user reviews")

display(polarizing_users_df[['game','developer','platform','genre','release_date','total_critics','total_users', 'metascore', 'user_score']])