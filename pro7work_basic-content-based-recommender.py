# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
books= pd.read_csv("../input/books.csv")

book_tags= pd.read_csv("../input/book_tags.csv")

rating= pd.read_csv("../input/ratings.csv")

tags=pd.read_csv("../input/tags.csv")

to_read=pd.read_csv("../input/to_read.csv")
#display all data

pd.set_option('display.max_colwidth', -1)
books.head()
#checking book_id and the best_book_id

books['check_ids']= np.where(books['book_id'] == books['best_book_id'],'0',np.nan)
books['check_ids'].isnull().values.any()


df_book= books[['book_id','books_count','isbn','authors','original_publication_year','title','language_code','average_rating','ratings_count','small_image_url']]
book_tags.rename(columns={'goodreads_book_id':'book_id'}, inplace=True)
book_tags=book_tags.merge(tags,on='tag_id',how='outer')
grouped= book_tags.groupby('book_id')['tag_name'].apply(' '.join)
df_book_new=pd.merge(df_book,grouped.to_frame(), on='book_id', how='inner')
df_book_new.head(3)
df_book_new.shape
#create metadata for similarity using Author,Tags and the language

def create_metadata(x):

    return ''.join(x['authors'])+'  '+''.join(x['tag_name'])+'  '+''.join(str(x['language_code']))



df_book_new['metadata']= df_book_new.apply(create_metadata,axis=1)
df_book_new['metadata']= df_book_new['metadata'].fillna('')
#finding the similarity between two books



from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer= TfidfVectorizer(stop_words='english')

vectorizer

df_book_matrix=vectorizer.fit_transform(df_book_new['metadata'])
df_book_matrix.shape
#cosine similarity using linear kernel



from sklearn.metrics.pairwise import linear_kernel



cos_matrix= linear_kernel(df_book_matrix, df_book_matrix)

cos_matrix
# 1D array for book title and indices



book_indices= pd.Series(df_book_new.index,index=df_book_new['title'])


def get_recommendations(name,sim):

   # indx=df_book_new.loc[df_book_new['title']==name].index

    indx=book_indices[name]

    sim_scores=list(enumerate(sim[indx]))

    new=sorted(sim_scores,key=lambda x: x[1],reverse=True)

   

    new=new[1:11]

    #print(new)

    book_idx=[x[0] for x in new]

    return (df_book_new['title'].iloc[book_idx])

    

    
get_recommendations('The Hunger Games (The Hunger Games, #1)',cos_matrix)