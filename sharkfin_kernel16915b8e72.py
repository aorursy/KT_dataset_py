credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})

movies_df_merge = movies_df.merge(credits_column_renamed, on='id')
movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
from sklearn.feature_extraction.text import TfidfVectorizer





tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3),

            stop_words = 'english')



# Filling NaNs with empty string

movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')
# Fitting the TF-IDF on the 'overview' text

tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])
from sklearn.metrics.pairwise import sigmoid_kernel



# Compute the sigmoid kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and movie titles

indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()
def give_rec(title, sig=sig):

    # Get the index corresponding to original_title

    idx = indices[title]



    # Get the pairwsie similarity scores 

    sig_scores1 = list(enumerate(sig[idx]))



    # Sort the movies 

    sig_scores = sorted(sig_scores1, key=lambda x: x[1], reverse=True)



    # Scores of the 10 most similar movies

    sig_scores = sig_scores[1:11]



    # Movie indices

    movie_indices = [i[0] for i in sig_scores]



    # Top 10 most similar movies

    return movies_cleaned_df['original_title'].iloc[movie_indices], sig_scores1
# Testing our content-based recommendation system with the seminal film Spy Kids

give_rec('Skyfall')
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")