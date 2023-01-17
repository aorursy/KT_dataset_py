import numpy as np

import pandas as pd

import spacy



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Load the large model to get the vectors

nlp = spacy.load('en_core_web_lg')

review_movies = pd.read_csv("../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv")
review_movies.head()
review_movies_1 = review_movies.copy()
# we convert all the rows where the genre has an 'unknown' to nan value

for idx, text in enumerate (review_movies_1.Genre): 

    if text == 'unknown': 

        review_movies_1.loc[:idx] = review_movies_1.drop([idx])

        
review_movies_2 = review_movies_1.loc[:,{'Genre', 'Plot'}]

review_movies_2
reviews = review_movies_2.dropna() # contain only the genre and plot features 



review_data = review_movies_1.dropna() # contrain the entire dataset 



print(reviews.isna().sum())

print('-'*40)

print(review_data.isna().sum())
reviews = review_data[:100]

with nlp.disable_pipes():

    vectors = np.array([nlp(review.Plot).vector for idx, review in reviews.iterrows()])

    

vectors.shape
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(vectors, reviews['Genre'], 

                                                    test_size=0.1, random_state=1)





model = LinearSVC(random_state=1, dual=False)



model.fit(X_train,y_train)





print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')
