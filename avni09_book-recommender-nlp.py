import pandas as pd

import numpy as np
books = pd.read_csv("../input/books.csv")
books.head(3)
books_sub = books[["book_id","title"]].copy()

books_sub
# Count Vectorizer - converts words into vectors.



from sklearn.feature_extraction.text import CountVectorizer



# initialize vectorizer

vect = CountVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english', min_df = 0.001)



vect.fit(books_sub['title'])

title_matrix = vect.transform(books_sub['title'])
title_matrix.shape
# Find vocabulary

features = vect.get_feature_names()

features
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_titles = cosine_similarity(title_matrix, title_matrix)

cosine_sim_titles.shape
title_id = 3

books_sub['title'].iloc[title_id]
top_n_idx = np.flip(np.argsort(cosine_sim_titles[title_id,])[-10:])

top_n_sim_values = cosine_sim_titles[title_id, top_n_idx]

top_n_sim_values
# find top n with values > 0

top_n_idx = top_n_idx[top_n_sim_values > 0]
# Matching books

books_sub['title'].iloc[top_n_idx]
# lets wrap the above code in a function

def return_sim_books(title_id, title_matrix, vectorizer, top_n = 10):

    

    # generate sim matrix

    sim_matrix = cosine_similarity(title_matrix, title_matrix)

    features = vectorizer.get_feature_names()



    top_n_idx = np.flip(np.argsort(sim_matrix[title_id,])[-top_n:])

    top_n_sim_values = sim_matrix[title_id, top_n_idx]

    

    # find top n with values > 0

    top_n_idx = top_n_idx[top_n_sim_values > 0]

    scores = top_n_sim_values[top_n_sim_values > 0]

    

    # find features from the vectorized matrix

    sim_books_idx = books_sub['title'].iloc[top_n_idx].index

    words = []

    for book_idx in sim_books_idx:

        try:

            feature_array = np.squeeze(title_matrix[book_idx,].toarray())

        except:

            feature_array = np.squeeze(title_matrix[book_idx,])

        idx = np.where(feature_array > 0)

        words.append([" , ".join([features[i] for i in idx[0]])])

        

    # collate results

    res = pd.DataFrame({"book_title" : books_sub['title'].iloc[title_id],

           "sim_books": books_sub['title'].iloc[top_n_idx].values,"words":words,

           "scores":scores}, columns = ["book_title","sim_books","scores","words"])

    

    

    return res

vect = CountVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english', min_df = 0.001)

vect.fit(books_sub['title'])

title_matrix = vect.transform(books_sub['title'])

return_sim_books(3,title_matrix,vect,top_n=20)
# Consider recommendation for the following book - 

#..."The Seven Spiritual Laws of Success: A Practical Guide to the Fulfillment of Your Dreams" ..



books[books["title"] == "The Seven Spiritual Laws of Success: A Practical Guide to the Fulfillment of Your Dreams"].index



return_sim_books(1854,title_matrix,vect,top_n=20)
# Number of times "guide" appears vs "success" in book titles



books_sub["title"].str.contains("\\bsuccess\\b", case=False).sum()
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english', min_df = 0.001)

vect.fit(books_sub['title'])

title_matrix = vect.transform(books_sub["title"])
import re

from nltk import word_tokenize          

from nltk.stem import PorterStemmer 

class PorterStem(object):

    def __init__(self):

        self.stm = PorterStemmer()

    def __call__(self, doc):

        return [self.stm.stem(t) for t in word_tokenize(doc) if re.match('[a-zA-Z0-9*.]',t)]



vect = TfidfVectorizer(tokenizer=PorterStem(),analyzer='word',ngram_range=(1,2),stop_words='english', min_df = 0.001)  
vect.fit(books_sub['title'])

title_matrix = vect.transform(books_sub["title"])

return_sim_books(1854,title_matrix,vect,top_n=20)
from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline
vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english', min_df = 0.001)



# Build the tfidf vectorizer from the training data ("fit"), and apply it 

# ("transform").

vect_matrix = vectorizer.fit_transform(books["title"])
vect_matrix.shape
# Apply SVD

svd_mod = TruncatedSVD(100)

lsa = make_pipeline(svd_mod, Normalizer(copy=False))



# Run SVD on the training data, then project the training data.

vect_matrix_svd = lsa.fit_transform(vect_matrix)
vect_matrix_svd.shape
return_sim_books(1854,vect_matrix_svd,vectorizer,top_n=20)
svd_comp = pd.DataFrame(svd_mod.components_)

svd_comp.columns = vectorizer.get_feature_names()



#svd_comp.to_csv("svd_components.csv")



svd_comp
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english', min_df = 0.0005)

vect.fit(books_sub['title'])



title_matrix = vect.fit_transform(books["title"])
title_matrix=title_matrix.toarray() 
title_matrix.shape
words = []

features = vect.get_feature_names()

for i in range(0,title_matrix.shape[0]):

    feature_array = np.squeeze(title_matrix[i,])

    idx = np.where(feature_array > 0)

    words.append([" , ".join([features[i] for i in idx[0]])])
books_sub["words"] = words
# To use use word2vec install gensim

# GloVe means Global Vectors for word representation. Is an unsupervised learning algorithm



from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec(glove_input_file="../input/gensim_glove.6B.50d.txt", word2vec_output_file="gensim_glove.6B.50d.txt")
from gensim.models.keyedvectors import KeyedVectors

glove_model = KeyedVectors.load_word2vec_format("../input/gensim_glove.6B.50d.txt", binary=False)
a = glove_model.word_vec("rich")

b = glove_model.word_vec("money")
from scipy.stats.stats import pearsonr



# Returns (Pearson’s correlation coefficient, 2-tailed p-value)

pearsonr(a,b)[0]
words = books_sub["words"].iloc[0][0].split(",")

words = [x.strip() for x in words]
words
chk = [glove_model.word_vec(x) for x in words if x in glove_model.vocab]
np.array(chk).mean(axis=0).shape


from tqdm import tqdm

title_vec = np.zeros((10000,50))

for i in tqdm(range(0,books_sub.shape[0])):

    words = books_sub["words"].iloc[i][0].split(",")

    words = [x.strip() for x in words]

    ind_word_vecs = [glove_model.word_vec(x) for x in words if x in glove_model.vocab]

    title_vec[i] = np.array(ind_word_vecs).mean(axis=0)
title_vec.shape
title_vec = np.nan_to_num(title_vec)
return_sim_books(1854,title_vec,vect,top_n=20)