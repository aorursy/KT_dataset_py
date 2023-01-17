# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import sigmoid_kernel
movies_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

movies_data.head()
#eliminating missing values.

print(pd.isnull(movies_data['overview']).sum())

movies_data['overview'] = movies_data['overview'].fillna('')

print(pd.isnull(movies_data['overview']).sum())
#generating tf-idf vectors for text document.

tfidf = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,3), stop_words='english')

tfidf_data = tfidf.fit_transform(movies_data['overview'])

tfidf_data
#getting similarity scores between all pairs of movie summaries.

sim_matrix = sigmoid_kernel(tfidf_data, tfidf_data)

#sample: for movie at index 0, similarity scores wrt every other movie.

sim_matrix[0]
#dict of movie titles and indices.

index_list = np.arange(0,movies_data.shape[0])

title_list = movies_data['original_title']

title2idx = dict(zip(title_list,index_list))
#function to generate recommendations.

def recommend_movie(current_title):

    for title,idx in title2idx.items():

        if title==current_title:

            current_idx = idx

    sim_scores = sim_matrix[current_idx]

    sim_scores = list(enumerate(sim_scores))

    sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    

    #getting top 10 recommendations based on similarity scores.

    top_similar_movies = sim_scores_sorted[1:11]

    

    print('Top Recommendations based on current movie: ',current_title)

    for i in top_similar_movies:

        for title, idx in title2idx.items():

            if i[0]==idx:

                print(i[0],title,'\n')

    
recommend_movie('Avatar')
recommend_movie('Spectre')
recommend_movie('The Dark Knight Rises')