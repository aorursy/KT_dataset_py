# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/kindle-books-dataset/Kindle_Book_Dataset.csv")

data.head()
#nlp_df=data.loc[:, ['title','author','description']]

nlp_df=data.iloc[:10001, [1,2,6]]

nlp_df.head()
nlp_df.info()
nlp_df.isnull().sum()
# Import the TfIdfVectorizer from scikit-learn library

from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF IDF Vectorizer Object

# with the removal of english stopwords turned on

tfidf=TfidfVectorizer(stop_words = 'english')

# Now costruct the TF-IDF Matrix by applying the fit_transform method on the description feature

tfidf_matrix = tfidf.fit_transform(nlp_df['description'])

print(tfidf_matrix)
from sklearn.metrics.pairwise import linear_kernel

cos_mat=linear_kernel(tfidf_matrix,tfidf_matrix)

cos_mat
print("shape of cos_mat",cos_mat.shape)
indices=pd.Series(nlp_df.index,index=nlp_df['title'])

indices
#got=nlp_df.index[nlp_df['title'] == 'Bloodfever: A Fever Novel [Kindle Edition]']

#got

def book_recommender(title,df=data,cos_mat=cos_mat,indices=indices):

    try:

        idx=indices[title]

    except KeyError:

        print("Book does not exist:(")

        return False

    #get the pairwise similarity score of all the books with that movie

    #and convert it into a list of tuples(position,similarity score)

    sim_scores=list(enumerate(cos_mat[idx]))

    sim_scores=sorted(sim_scores,key=lambda x: x[1],reverse =True)

    # Get the scores of top 10 most similar books.Ignore the first book

    sim_scores=sim_scores[1:11]

    # Get the Book indices

    book_indices=[sim_score[0] for sim_score in sim_scores]

    # Return the top 10 similar books

    return data['title'].iloc[book_indices]



    