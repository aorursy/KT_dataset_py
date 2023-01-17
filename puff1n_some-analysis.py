import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/movie_metadata.csv')

df = df.drop(df[df.budget > 2000000000].index) #drop the rows where budget greater than 2 000 000 000
df['profit'] = df.gross - df.budget

top_15_fails = df.sort(columns='profit').head(15)

top_15_success = df.sort(columns='profit', ascending=False).head(15)

top_15_fails[['gross', 'budget']].groupby(df['movie_title']).sum().plot.barh(stacked=True, 

                                                                            title='Top-15 fails')
top_15_success[['budget', 'gross']].groupby(df['movie_title']).sum().plot.barh(stacked=True, 

                                                                              title='Top-15 success')
fail_country = df.sort(columns='profit').head(100).groupby(df['country']).sum()

fail_country['profit'].plot.bar()
success_country = df.sort(columns='profit', ascending=False).head(100).groupby(df['country']).sum()

success_country['profit'].plot.bar()
x = df.movie_facebook_likes.groupby(df.country).sum().plot.bar(figsize=(13, 4), title='Likes by country')
x = df.movie_facebook_likes.groupby(df.title_year).sum().plot.bar(figsize=(12, 4), ylim=(0, 6500000), 

                                                                  title='Likes by year')
df.movie_facebook_likes.groupby(df.language).sum().plot.bar(figsize = (12, 4), title='Likes by language')
df.budget.groupby(df.country).mean().plot.bar(figsize = (12, 5), ylim=(0, 240000000), 

                                             title='Mean budget by country')

#But why the Thailand and Japan
max_facebook_likes_index = df.movie_facebook_likes.argmax()

#max_facebook_likes_index

df.movie_facebook_likes.max()

df[93:94].movie_title


fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(df.duration, df.movie_facebook_likes)

ax.annotate('Interstellar', xy=(1, 1), xytext=(175, 350000))
mean_imdb = df.imdb_score.groupby(df.director_name).mean().sort_values(ascending=False)

mean_imdb[:15].sort_values().plot.barh(figsize=(6, 8), title='Top-15 directors with highest imdb score')
from sklearn import cross_validation

from sklearn.linear_model import LinearRegression
df.dropna(inplace=True)

X = df[['num_critic_for_reviews', 'duration',

       'director_facebook_likes', 'actor_3_facebook_likes',

       'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes', 

        'facenumber_in_poster', 'num_user_for_reviews', 'budget', 'actor_2_facebook_likes',

        'movie_facebook_likes']]

y = df[['imdb_score']]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)
lr = LinearRegression()

lr.fit(X_train, y_train)

print(lr.score(X_train, y_train))

predicted = lr.predict(X_test)

plt.scatter(y_test, predicted, color='gray')