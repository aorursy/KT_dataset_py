import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import io

import math
df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

df.head()
print(df.columns)
original_len = len(df)

dropped_df = df.dropna()

dropped_len = len(dropped_df)

print("% of rows with missing values: " + str((original_len - dropped_len) / dropped_len * 100) + '%')

print()

print("Number of null values in each column: ")

print(df.isnull().sum())

df = dropped_df
df.drop(columns=['show_id', 'date_added', 'duration', 'description'], inplace=True)
df['country'] = df['country'].map(lambda x: x.split(',')[0])



bag_of_words_data = ['director', 'cast', 'listed_in']



for col in bag_of_words_data:

    df[col] = df[col].map(lambda x : x.lower().replace(' ', '').split(',')[:3])



df['bag_of_words'] = ''

for i, row in df.iterrows():

    words = [' '.join(row[col]) for col in bag_of_words_data]

    df.loc[i, 'bag_of_words'] = ' '.join(words)

    

df.drop(columns=bag_of_words_data, inplace=True)

df.head()
df = pd.get_dummies(df, columns=['type', 'rating', 'country'])

df.reset_index(drop=True, inplace=True)
vectorizer = CountVectorizer()

count_matrix = vectorizer.fit_transform(df['bag_of_words'])



similarities = cosine_similarity(count_matrix, count_matrix)

dissimilarities = 1 - similarities
def euclidean_distance(row1, row2, release_year_weighting=0.1):

    row1_features = np.array([row1[col] for col in df.columns if col != 'title' and col != 'bag_of_words' and col != 'release_year'])

    row1_features = row1_features.astype(np.int16)

    row2_features = np.array([row2[col] for col in df.columns if col != 'title' and col != 'bag_of_words' and col != 'release_year'])

    row2_features = row2_features.astype(np.int16)

    diffs = np.subtract(row1_features, row2_features)



    release_year_diff = release_year_weighting * (row1['release_year'] - row2['release_year'])

    diffs = np.append(diffs, [release_year_diff])



    return math.sqrt(np.sum([diff ** 2 for diff in diffs]))



def total_distance(row1_index, row2_index, bag_of_words_weighting=2):

    row1 = df.iloc[row1_index]

    row2 = df.iloc[row2_index]

    distance = euclidean_distance(row1, row2)

    distance += bag_of_words_weighting * dissimilarities[row1_index][row2_index]

    return distance
def recommend_for(title, num_recommendations=10):

    all_titles = df['title']

    title_instances = all_titles[all_titles == title]

    if title_instances.empty:

        print("Sorry! We can't seem to find that movie in our collection")

        return

    curr_index = all_titles[all_titles == title].index[0]



    distances = list()

    for i, row in df.iterrows():

        distances.append((row, total_distance(curr_index, i)))

    distances.sort(key=lambda tup: tup[1])

    results = list(map(lambda tup: tup[0]['title'], distances[1:num_recommendations+1]))

    print("After watching " + title + ", we recommend: ")

    for res in results:

        print(res)

    print()
recommend_for("The Battle of Midway")

recommend_for("You")

recommend_for("The Perfect Date")