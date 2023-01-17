import os
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from string import punctuation
from nltk.corpus import stopwords
metadata = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', low_memory=False)
metadata.head()
metadata.shape
abstracts = metadata.loc[:,('pubmed_id','abstract')]
abstracts.dropna(how='any', inplace=True)
abstracts.head()
abstracts.shape
nlp = spacy.load('en_core_web_lg')
%%time
# Convert every Abstract to a Spacy Object using the pretrained model. 
# This will transform each abstract on a vector of 300 dimensions, among other characteristics.
# To save time and computation requirements, we disable all other pipes of the model.
with nlp.disable_pipes(nlp.pipe_names):
    abstracts['spacy_docs'] = [nlp(row) for row in abstracts['abstract']]
query = 'diagnostics and surveillance'
#Requesting the vector similarity between a given query and each of the abstracts.
doc_query = nlp(query)
abstracts['semantic_similarity'] = [doc_query.similarity(row) for row in abstracts['spacy_docs']]
semantic_similarity = abstracts[['pubmed_id', 'abstract', 'semantic_similarity']].copy()
semantic_similarity.sort_values(by='semantic_similarity', axis=0, ascending=False, inplace=True)
semantic_similarity.head()
#1st result
abstracts['abstract'].loc[701]
#2nd result
abstracts['abstract'].loc[39332]
#3rd result
abstracts['abstract'].loc[19188]
vectors = np.array([doc.vector for doc in abstracts['spacy_docs']])
#saving vectors to Notebook's output
np.savetxt('abstracts_vectors.gz', vectors)
#Loading vectors from Notebook's output
vectors_loaded = np.loadtxt('abstracts_vectors.gz')
#Center the vectors by subtracting the global mean. 
vec_mean = vectors_loaded.mean(axis=0)
vec_centered = vectors_loaded - vec_mean
query = "diagnostics and surveillance"
#Vectorizing the query and centering it by subtracting the global mean.
vec_query = nlp(query).vector
cent_vec_query = vec_query - vec_mean
#Cosine Similarity: Instead of using sklearn method, that would require reshaping the vectors numpy, I'm defining the function manually: 
def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

cos_similarity = np.array([cosine_similarity(cent_vec_query, doc) for doc in vec_centered])
abstracts['cosine_similarity'] = cos_similarity
cosine_similarity = abstracts[['pubmed_id', 'abstract', 'cosine_similarity']].copy()
cosine_similarity.sort_values(by='cosine_similarity', axis=0, ascending=False, inplace=True)
cosine_similarity.head()
#1st result (The same as in method 1)
abstracts['abstract'].loc[701]
#2nd result
abstracts['abstract'].loc[8816]
#3rd result
abstracts['abstract'].loc[50944]
#Create a new column with the words in the abstract, removing punctuation and stopwords from nltk.

def Preprocessing(cell):
    stop = set(stopwords.words('english'))
    stop.add('abstract')
    words_list = str(cell).translate(str.maketrans('', '', punctuation)).lower().split()
    return list(filter(lambda word: word not in stop, words_list))


abstracts['tfidf_tokenized_text'] = abstracts['abstract'].apply(lambda x: " ".join(Preprocessing(x)))
abstracts.head()
#Train a Tfidf model using the previously tokenized text
texts = abstracts['tfidf_tokenized_text']
tf = TfidfVectorizer(min_df=1)
tf = tf.fit(texts)
##Using the trained Tf-idf model, transform the texts. This produces a sparse matrix with a line per document,
# and one column per token in the vocabulary of the model. In this case, 135,909 columns. The value of each cell registers
# the frequency of each token in the document.
trx_text = tf.transform(texts)
trx_text.shape
#Normalize the vectors 
centroids = normalize(trx_text, copy=False)
query = 'diagnostics and surveillance'
query_processed = Preprocessing(query)
query_trans = tf.transform(query_processed)
query_cent = normalize(query_trans)
sorting = linear_kernel(query_cent, centroids)
index = np.argsort(sorting[0], axis=-1)[::-1]
#In this case, what we obtain from the linear kernel is a sorted list of indexes from the dataframe. 
# Therefore, I am sorting the dataframe using that index, and creating a rank column based on that order.
abstracts = abstracts.iloc[list(index)].copy()
abstracts['tfidf_rank'] = list(range(1, abstracts.shape[0]+1))
abstracts.head()
#1st result
abstracts['abstract'].loc[45631]
#2nd result
abstracts['abstract'].loc[13504]
#3rd result
abstracts['abstract'].loc[10340]
abstracts.sort_values(by='cosine_similarity', axis=0, ascending=False, inplace=True)
abstracts['cosine_rank'] = list(range(1, abstracts.shape[0]+1))
abstracts.sort_values(by='semantic_similarity', axis=0, ascending=False, inplace=True)
abstracts['semantic_rank'] = list(range(1, abstracts.shape[0]+1))
docs_num, _ = abstracts.shape
abstracts['global_rank'] = abs((abstracts['tfidf_rank'] - docs_num) + (abstracts['cosine_rank'] - docs_num) + (abstracts['semantic_rank'] - docs_num))
abstracts.sort_values(by='global_rank', ascending=False, inplace=True)
abstracts.head()
#1st result
abstracts['abstract'].loc[21183]
#2nd result
abstracts['abstract'].loc[12340]
#3rd result
abstracts['abstract'].loc[14411]
metadata = metadata.merge(abstracts[['global_rank']], how='left', left_index=True, right_index=True)
metadata.to_csv('metadata_abstracts_ranked.csv')