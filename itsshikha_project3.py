import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
%matplotlib inline
import nltk
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer, word_tokenize
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


corpus = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
corpus = corpus[['Review Text', 'Clothing ID']]
# We have selected Clothing ID 1080 for oyr analysis
corpus = corpus.loc[corpus['Clothing ID'] == 1080 ]
corpus = corpus['Review Text']
#  Dropping the entries with empty reviews
corpus.dropna(inplace = True)
# Tokenizing the corpus
sent_tokenized_corpus = []
# To store the reviews
reviewsList = []
# Firstly we are tokenizing reviews using sentence tokenizer
for review in corpus :
    reviewsList.append(review)
    sent_tokenized_corpus.append(sent_tokenize(review))
    
word_tokenized_reviews = []
words = []
# Now we are using word tokenizer to tokenize the review into words
for review in sent_tokenized_corpus :
    for sent in review :
        words += (word_tokenize(sent))
    word_tokenized_reviews.append(words)
    words = []
#print(word_tokenized_reviews)

lemmatizer = WordNetLemmatizer()

review_str = ""
stop_words = set(stopwords.words('english'))

final_corpus = []
for review in word_tokenized_reviews:
    for words in review :
        if words not in stop_words:
            review_str += (" "+(lemmatizer.lemmatize(words.lower())))
    final_corpus.append(review_str)
    review_str =""

# Tokenizing the corpus after removing stop words and lemmetizing
vectorizer = CountVectorizer(min_df=0, stop_words=stop_words)

docs_tf = vectorizer.fit_transform(final_corpus)
vocabulary_terms = vectorizer.get_feature_names()

#selecting the keywords
keywords = ["love", "pretty", "incredible", "adorable", "stunner" ]


docs_query_tf = vectorizer.transform(final_corpus + [' '.join(keywords)]) 

transformer = TfidfTransformer(smooth_idf = False)
tfidf = transformer.fit_transform(docs_query_tf.toarray())

# D x V document-term matrix 
tfidf_matrix = tfidf.toarray()[:-1] 

# 1 x V query-term vector 
query_tfidf = tfidf.toarray()[-1] 
query_doc_tfidf_cos_dist = [cosine(query_tfidf, doc_tfidf) for doc_tfidf in tfidf_matrix]
query_doc_tfidf_sort_index = np.argsort(np.array(query_doc_tfidf_cos_dist))

for rank, sort_index in enumerate(query_doc_tfidf_sort_index):
    if rank == 5 :
        break
    print("The rank is", rank)
    print("The cosine distance is", query_doc_tfidf_cos_dist[sort_index])
    print("Review")
    print(reviewsList[sort_index])
    
tf_matrix = docs_tf.toarray() # D x V matrix 
A = tf_matrix.T 

U, s, V = np.linalg.svd(A, full_matrices=1, compute_uv=1)
K = 2 # number of components

A_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), V[:K, :])) # D x V matrix 

docs_rep = np.dot(np.diag(s[:K]), V[:K, :]).T # D x K matrix 
terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # V x K matrix 

key_word_indices = [vocabulary_terms.index(key_word) for key_word in keywords] # vocabulary indices 

key_words_rep = terms_rep[key_word_indices,:]     
query_rep = np.sum(key_words_rep, axis = 0)

query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

for rank, sort_index in enumerate(query_doc_sort_index):
    if rank == 5 :
        break
    print("The rank is", rank)
    print("The cosine distance is", query_doc_tfidf_cos_dist[sort_index])
    print("Review")
    print(reviewsList[sort_index])
# plot documents in the new space  
%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(docs_rep[:,0], docs_rep[:,1], c=query_doc_cos_dist) # all documents 
plt.scatter(query_rep[0],query_rep[1],   marker='+', c='red') # the query 
plt.xlabel("Component 1")
plt.ylabel("Component 2")