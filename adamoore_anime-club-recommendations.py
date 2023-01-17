# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# TODO: Use average review scores to further hone results.
# TODO: Add genres to the output CSV
# TODO: Filter out all hentai - we won't watch them.
# TODO: Filter out all music videos - we won't watch them.
# TODO: Add episode count to output file. 13 episode long series fit better into a single semester of viewing.
# TODO: Distinguish between shorts and full-length episodes. Shorts have a much lower average audience score on MAL, but we're willing to spend 1-2 minutes watching something with a relatively low score.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import itertools
import collections
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
anime = pd.read_csv('../input/anime-recommendations-database/anime.csv') # Data from MyAnimeList
ratings = pd.read_csv('../input/real-animu/anime-club-rankings.csv') # Averaged review scores of anime from the club's MyAnimeList account

# Genres are given as a comma separated list
# We want to turn that into individual columns where
# a value of 1 means that the anime is tagged with that genre
# and 0 means the anime is not tagged with that genre.
anime['genre'] = anime['genre'].fillna('None') # filling 'empty' data. Some anime have no associated genres. We still want to make predictions about them though.
anime['genre'] = anime['genre'].apply(lambda x: x.split(', ')) # split genre into list of individual genre
genre_data = itertools.chain(*anime['genre'].values.tolist()) # flatten the list
genre_counter = collections.Counter(genre_data)
genres = pd.DataFrame.from_dict(genre_counter, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})
genres.sort_values('count', ascending=False, inplace=True)
genre_map = {genre: idx for idx, genre in enumerate(genre_counter.keys())}
def extract_feature(genre):
    feature = np.zeros(len(genre_map.keys()), dtype=int)
    feature[[genre_map[idx] for idx in genre]] += 1
    return feature
anime['genre'] = anime['genre'].apply(lambda x: extract_feature(x))

# We want to create a dataframe with our training data - this will only have anime the club has reviewed.
x = pd.merge(anime, ratings, on='name') # Bring in the review scores
x.dropna(subset=['score'], inplace=True) # Drop everything we haven't scored from the dataframe
X = pd.DataFrame(x['genre'].values.tolist(), columns=genres.genre, index=x.name) # Create our training dataframe using just the anime name and genres
x.set_index('name', inplace=True)
y = pd.DataFrame(x.score) # Create the series of known review scores indexed by anime name.

# Set up a pipeline we can apply to both our training data and our unlabeled data.
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_pipeline = make_pipeline(SimpleImputer(), my_model)

# Cross validate our learning model by testing it on known values
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
avg_mae = -scores.mean()
print("Average mean absolute error: " + str(avg_mae))
# Now we're going to create a new dataframe with just unlabeled data.
test_X = pd.merge(anime, ratings, on='name', how='left')
test_X = test_X[pd.isnull(test_X['score'])]
test_X = pd.DataFrame(test_X['genre'].values.tolist(), columns=genres.genre, index=test_X.name)

# Now to use our trained model to make predictions for unknown values
my_pipeline.fit(X, y) # Train the model by running the known data through our pipeline.
predicted_scores = my_pipeline.predict(test_X) # Use the model to make predictions by running our unknown data through the pipeline

# Output our predictions to a csv
predictions = pd.DataFrame({'Name': test_X.index, 'Score': predicted_scores})
predictions.to_csv('predictions.csv')