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
users=pd.read_csv("/kaggle/input/post-recommender/users.csv")

users=users.rename(columns={"_id":"user_id"})

users.head()
post_df=pd.read_csv("/kaggle/input/post-recommender/posts.csv")

post_df=post_df.rename(columns={"_id":"post_id"})

post_df.head()
views=pd.read_csv("/kaggle/input/post-recommender/views.csv")

views.head()
from sklearn.feature_extraction.text import TfidfVectorizer



TfidfVectorizer

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3),

            stop_words = 'english')



# Filling NaNs with empty string

post_df['category'] = post_df['category'].fillna('')
tfv_matrix = tfv.fit_transform(post_df['category'])

tfv_matrix.shape
from sklearn.metrics.pairwise import sigmoid_kernel



sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(post_df.index, index=post_df['title']).drop_duplicates()
indices.head()
def give_rec(title,no_of_rec, sig=sig):

    # Get the index corresponding to original_title

    idx = indices[title]



    # Get the pairwsie similarity scores 

    sig_scores = list(enumerate(sig[idx]))



    # Sort the movies 

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)



    # Scores of the 10 most similar movies

    sig_scores = sig_scores[0:no_of_rec+1]



    # post indices

    post_index = [i[0] for i in sig_scores]



    # Top 10 most similar movies

    return post_df['title'].iloc[post_index]
def recommend_post(title,n=10):

  rec=give_rec(title,n).to_list()

  count=0

  for item in rec:

    if not (item==title) and count<=n:

      count+=1

      print(item)
data_df = pd.merge( users,views, on='user_id')

data_df = pd.merge( data_df,post_df, on='post_id') 

data_df=data_df.drop(columns=['user_id','gender','academics','post_id','category'])

data_df.head()
def recommend_user(user_name,n=10):

  curr=data_df.query('name == @user_name')

  latest=max(curr['timestamp'])

  curr=curr.query('timestamp==@latest')

  for i in (curr['title']):

    print("Recent Post viewed:-", i)

    print("\nRecommended for you:\n")

    recommend_post(i,n)

recommend_post("Ml and AI",5)
recommend_user("Sahana B")