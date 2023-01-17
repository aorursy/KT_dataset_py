import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Necessary Dependancies



import pandas as pd
# Load the dataset

data = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', low_memory=False)

data.head()
data.shape
# Calculate C

C = data['vote_average'].mean()

print(C)
# Calculate m

m = data['vote_count'].quantile(0.90)

print(m)
# Filter out all the qualified movies into a new DF

movies = data.copy().loc[data['vote_count'] >= m]

movies.shape
# Function to compute the weighted rating of each movie

def weighted_rating(x,m=m,C=C):

    v = x['vote_count']

    R = x['vote_average']

    # IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`

movies['score'] = movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above

movies = movies.sort_values('score', ascending=False)
#Print the top 15 movies

movies[['title', 'vote_count', 'vote_average', 'score']].head(15)