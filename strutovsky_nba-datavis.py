import numpy as np 

import pandas as pd 

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw_data = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')

raw_data.head()
raw_data.info()
pd.DataFrame(raw_data.isna().sum(), columns=['how much are NaN types'])
cleaned_data = raw_data.copy()



cleaned_data['jersey'] = cleaned_data['jersey'].apply(lambda x: int(x[1:])) # delete '#' symbol

cleaned_data['team'] = cleaned_data['team'].fillna('no team')   # fill all n/a with 'no team' string

cleaned_data['height'] = cleaned_data['height'].apply(lambda x: float(x[2+x.find('/'):])) # convert to meters

cleaned_data['weight'] = cleaned_data['weight'].apply(lambda x: float(x[2+x.find('/'):-4])) # convert to kg

cleaned_data['salary'] = cleaned_data['salary'].apply(lambda x: int(x[1:])) # delete '#' symbol

cleaned_data['draft_round'] = cleaned_data['draft_round'].apply(lambda x: int(x) if x.isdigit() else 0)

cleaned_data['draft_peak'] = cleaned_data['draft_peak'].apply(lambda x: int(x) if x.isdigit() else 0)

cleaned_data['college'] = cleaned_data['college'].fillna('no college')

cleaned_data['experience_years'] = 2020 - cleaned_data['draft_year']

cleaned_data = cleaned_data.drop(['draft_year'], axis=1)
# change bday on age

cleaned_data['b_day'] = cleaned_data['b_day'].apply(lambda x: x[-2:])

cleaned_data['b_day'] = cleaned_data['b_day'].apply(lambda x: int('20'+x) if x[0] == '0' else int('19'+x))

cleaned_data['age'] = 2020 - cleaned_data['b_day']

cleaned_data = cleaned_data.drop(['b_day'], axis=1)
cleaned_data
plt.figure(figsize=(15,8))

sns.heatmap(cleaned_data.corr(), annot=True, linewidths=0.5, linecolor='black', cmap='coolwarm')

plt.show()
plt.figure(figsize=(10,8))

sns.set_style("whitegrid")



ax = sns.regplot(x='salary', y='rating', data=cleaned_data, color='black')

ax.set_title("Regression plot of player's rating and salary", fontsize=20)

ax.set_xlabel('salary', fontsize=15)

ax.set_ylabel('rating', fontsize=15)



plt.show()
plt.figure(figsize=(15,10))

bins = np.arange(12,34) - 0.5



plt.hist((cleaned_data['age'] - cleaned_data['experience_years']), bins=bins)

plt.title("Player age when has been started his career", fontsize=20)

plt.xticks(range(12,33))

plt.yticks(range(0, 111, 10))

plt.xlabel("Age", fontsize=15)

plt.ylabel("Count of players", fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

ax = sns.barplot(data = cleaned_data, x = 'experience_years', y = 'salary')

plt.xlabel('Years after draft', fontsize=15)

plt.ylabel('Salary', fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

ax = sns.countplot(data=cleaned_data, x = 'country', order = cleaned_data['country'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=13)

ax.set(xlabel='', ylabel='count')

plt.show()
plt.figure(figsize=(15,10))

top20_rated = pd.DataFrame(cleaned_data[['rating', 'full_name']]).sort_values(by='rating', ascending=False)[:20]

ax = sns.barplot(data=top20_rated, x = 'rating', y = 'full_name')

ax.set(ylabel='')

ax.set_title("Top 20 players by rating", fontsize=15)

plt.show()
plt.figure(figsize=(15,10))

ax = sns.countplot(data=cleaned_data, x = 'position', order = cleaned_data['position'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=13)

ax.set(xlabel='', ylabel='count')

ax.set_title("Most playable positions", fontsize=15)

plt.show()