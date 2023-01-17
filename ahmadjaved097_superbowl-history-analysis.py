import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# setting plot style for all the plots

plt.style.use('fivethirtyeight')
df = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')

df.head()
print('Number of rows in the dataset: ', df.shape[0])

print('Number of columns in the dataset: ', df.shape[1])
df.info()
df['Date'] = pd.to_datetime(df['Date'])

df['Date'] = df['Date'].dt.year
df.describe().round(decimals=3).drop('Date', axis=1)
fig, ax = plt.subplots(figsize=(18,6))

ax.plot(df['Date'], df['Winner Pts'], marker='.', mew=5, color='dodgerblue', label='Winners')

ax.plot(df['Date'],df['Loser Pts'], marker='+', color='red', label='Losers', mew=3)

ax.set_xlabel('Years')

ax.set_ylabel('Points Scored')

ax.set_title('Points scored by Winning and Loosing Teams over the Years')

ax.legend()

plt.show()
df[df['Winner Pts'] == df['Winner Pts'].max()][['Winner', 'Winner Pts', 'Date']]
df[df['Winner Pts'] == df['Winner Pts'].min()][['Winner', 'Winner Pts', 'Date']]
df[df['Loser Pts'] == df['Loser Pts'].max()][['Loser', 'Loser Pts', 'Date']]
df[df['Loser Pts'] == df['Loser Pts'].min()][['Loser', 'Loser Pts', 'Date']]
# this dataframe contains the winning count of each team

winning_count = pd.DataFrame(df['Winner'].value_counts()).reset_index()

winning_count.index += 1

winning_count.rename(columns = {

    'index':'Team Name',

    'Winner':'Count'

}, inplace=True)



winning_count.sort_values(by='Count', ascending=False, inplace=True)
plt.figure(figsize=(20,7))

sns.barplot(y='Team Name', x='Count', data=winning_count,

           edgecolor='black',

           linewidth=2)

plt.title('Number of times each team has won throughout the years')

plt.xticks(rotation=90)

plt.show()
losing_count = pd.DataFrame(df['Loser'].value_counts()).reset_index()

losing_count.index += 1

losing_count.rename(columns = {

    'index':'Team Name',

    'Loser':'Count'

}, inplace=True)



losing_count.sort_values(by='Count', ascending=False, inplace=True)
plt.figure(figsize=(20,7))

sns.barplot(x='Team Name', y='Count', data=losing_count,

           edgecolor='black',

           linewidth=2, palette='Blues_r')

plt.title('Number of times each team has won throughout the years')

plt.xticks(rotation=90)

plt.show()
df['Winning Margin'] = df['Winner Pts'] - df['Loser Pts']
plt.figure(figsize=(18,6))

plt.plot(df['Date'],df['Winning Margin'], marker='.', mew=3, linewidth=4,mec='black', color='dodgerblue')

plt.xlabel('Years')

plt.ylabel('Winning Margin')

plt.title('Winning Margin over the years')

plt.show()
df['Match'] = df['Winner'] + ' vs ' + df['Loser']

df2 = df.sort_values(by='Winning Margin', ascending=False)

df2 = df2.head(10)
plt.figure(figsize=(18,7))

sns.barplot(y='Match', x='Winning Margin', data=df2,

            edgecolor='black',linewidth=2)

plt.title('Top 10 teams by winning point margin')

plt.show()
state_count = pd.DataFrame(df['State'].value_counts()).reset_index()

state_count.index += 1

state_count.rename(columns = {

    'index':'State',

    'State':'Count'

}, inplace=True)



state_count.sort_values(by='Count', ascending=False, inplace=True)
plt.figure(figsize=(18,6))

g = sns.barplot(y='State', x='Count', data=state_count, edgecolor='black', linewidth=2)

g.set_title('Statewise Number of Matches', y=1.05)

g.set(xlabel='States', ylabel='Number of Matches held')

plt.show()
city_count = pd.DataFrame(df['City'].value_counts()).reset_index()

city_count.index += 1

city_count.rename(columns = {

    'index':'City',

    'City':'Count'

}, inplace=True)



city_count.sort_values(by='Count', ascending=False, inplace=True)

city_count = city_count.head()
plt.figure(figsize=(18,6))

g = sns.barplot(y='City', x='Count', data=city_count, edgecolor='black',linewidth=2)

g.set_title('Top 5 cities in terms of number of matches held', y=1.05)

g.set(xlabel='Cities', ylabel='Number of Matches held')

plt.show()
mvp_count = pd.DataFrame(df['MVP'].value_counts()).reset_index()

mvp_count.index += 1

mvp_count.rename(columns = {

    'index':'MVP',

    'MVP':'Count'

}, inplace=True)



mvp_count.sort_values(by='Count', ascending=False, inplace=True)

mvp_count = mvp_count.head()
plt.figure(figsize=(18,6))

g = sns.barplot(y='MVP', x='Count', data=mvp_count, edgecolor='black',linewidth=2)

g.set_title('Top 5 Most Valuable Players', y=1.05)

g.set(xlabel='MVP', ylabel='Count')

plt.show()
stadium_count = pd.DataFrame(df['Stadium'].value_counts()).reset_index()

stadium_count.index += 1

stadium_count.rename(columns = {

    'index':'Stadium',

    'Stadium':'Count'

}, inplace=True)



stadium_count.sort_values(by='Count', ascending=False, inplace=True)

stadium_count = stadium_count.head(4)
plt.figure(figsize=(14,4))

g = sns.barplot(y='Stadium', x='Count', data=stadium_count, edgecolor='black',linewidth=2)

g.set_title('Stadiums where most number of matches were held', y=1.05)

g.set(xlabel='Stadium Names', ylabel='Count')

plt.show()