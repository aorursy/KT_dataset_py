import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')

df.head()
df.tail()
print('Number of rows in the dataset: ', df.shape[0])

print('Number of columns in the dataset: ', df.shape[1])
null_values = df.isnull().sum()

null_values = pd.DataFrame(null_values, columns=['Missing Values'])

null_values
total_unique_categories = list(range(df['category'].nunique()))

print('------CATEGORIES----------')

for category in zip(total_unique_categories, df['category'].unique()):

    print(category[0], ': ', category[1])
print('Number of unique nominees for the award: ', df['nominee'].nunique())
print('Number of unique films in the award ceremony: ', df['film'].nunique())
nominee_most_awards = df.groupby('nominee')['win'].sum()

nominee_most_awards = pd.DataFrame(nominee_most_awards[nominee_most_awards == nominee_most_awards.max()])

nominee_most_awards
most_nominated_nominee = df.groupby('nominee')['win'].count()

pd.DataFrame(most_nominated_nominee[most_nominated_nominee == most_nominated_nominee.max()])
more_5_win = df.groupby('nominee')['win'].sum()

more_5_win = more_5_win.reset_index()

more_5_win = more_5_win[more_5_win['win'] >= 5].sort_values(ascending=False, by='win')



# print('There are ', more_5_win['win'].count(), 'nominees who won more than 5 awards')

plt.figure(figsize=(20,8))

sns.set_style('whitegrid')

sns.barplot(x='nominee', y='win', data=more_5_win, palette='hls')

plt.title('Nominees who won more than 5 awards', fontsize=14)

plt.xlabel('Nominee', fontsize=14)

plt.ylabel('Total Wins', fontsize=14)

plt.xticks(fontsize=12, rotation=90)

plt.yticks(fontsize=12)

plt.show()
more_15_nominated = df.groupby('nominee')['win'].count()

more_15_nominated = more_15_nominated.reset_index()

more_15_nominated = more_15_nominated[more_15_nominated['win'] >= 15].sort_values(ascending=False, by='win')



# top_10_nominated

plt.figure(figsize=(20,8))

sns.set_style('whitegrid')

sns.barplot(x='nominee', y='win', data=more_15_nominated, palette='hls')

plt.title('Nominees who were nominated more than 15 times', fontsize=14)

plt.xlabel('Nominee', fontsize=14)

plt.ylabel('Nomination Count', fontsize=14)

plt.xticks(fontsize=12, rotation=90)

plt.yticks(fontsize=12)

plt.show()
film_most_awards = df.groupby('film')['win'].sum()

pd.DataFrame(film_most_awards[film_most_awards == film_most_awards.max()])
more_5_films = df.groupby('film')['win'].sum()

more_5_films = more_5_films.reset_index()

more_5_films = more_5_films[more_5_films['win'] >= 5].sort_values(ascending=False, by='win')



plt.figure(figsize=(20,8))

sns.set_style('whitegrid')

sns.barplot(x='film', y='win', data=more_5_films, palette='hls')

plt.title('Films that have won atleast 5 awards', fontsize=12)

plt.xlabel('Film', fontsize=12)



plt.ylabel('Award Count', fontsize=12)

plt.xticks(rotation=90, fontsize=12)

plt.yticks(rotation=90, fontsize=12)

plt.show()
film_awards_year = df.groupby(['film', 'year_award'])['win'].sum()

pd.DataFrame(film_awards_year[film_awards_year == film_awards_year.max()])
film_award_df = pd.DataFrame(film_awards_year).reset_index()

film_four_awards = film_award_df[film_award_df['win'] >= 4].sort_values(ascending = False, by='win')





plt.figure(figsize=(20,8))

sns.set_style('whitegrid')

sns.barplot(x='film', y='win', data=film_four_awards, palette='hls')

plt.title('Films with atleast 4 awards in a single year', fontsize=12)

plt.xlabel('Film', fontsize=12)

plt.ylabel('Award Count', fontsize=12)

plt.xticks(rotation=90, fontsize=12)

plt.yticks(rotation=90, fontsize=12)

plt.show()
film_nominations = df.groupby(['film', 'year_award'])['win'].count()

pd.DataFrame(film_nominations[film_nominations == film_nominations.max()]).reset_index()
director_motion_picture = df[df['category'].str.contains('Best Director - Motion Picture')]

director_motion_picture = director_motion_picture.groupby('nominee')['win'].count().reset_index().sort_values(ascending=False, by='win')

director_motion_picture = director_motion_picture[director_motion_picture['win'] >= 5]

# director_motion_picture



plt.figure(figsize=(15,6))

sns.set_style('whitegrid')

sns.barplot(x='nominee', y='win', data=director_motion_picture, palette='hls')

plt.title('Most nominated directors in motion picture category', fontsize=12)

plt.xlabel('Director', fontsize=12)



plt.ylabel('Nominations', fontsize=12)

plt.xticks(rotation=90, fontsize=12)

plt.yticks(rotation=90, fontsize=12)

plt.show()
winner_director = df[df['category'].str.contains('Best Director - Motion Picture')]

winner_director = winner_director.groupby('nominee')['win'].sum()

pd.DataFrame(winner_director[winner_director == winner_director.max()])