# Global variables 
dataset_filename = "../input/Womens Clothing E-Commerce Reviews.csv"
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
from numpy import array
from sklearn.decomposition import TruncatedSVD

%matplotlib inline
# Loading dataset
data = pd.read_csv(dataset_filename, index_col=0)
# Looking at some of the top rows of dataset
data.head()
# Description
data.describe()
# Info
data.info()
# Printing the unique Clothing IDs along with their frequencies
data['Clothing ID'].value_counts()[:5]
# From here, we can conclude Clothing ID 1078 as most common. So, we will be using this for the rest of our project
datax = data.loc[data['Clothing ID'] == 1078 , :] # We will be calling this data as datax
datax.head()
datax.info()
corpus = [review for (id,review) in datax['Review Text'].iteritems() if isinstance(review,str)]
# Creating dictionary of review to id
review_to_id_dict = {review : id for (id,review) in enumerate(corpus)}
corpus_tokenized = np.array([review.split() for review in corpus])

print(corpus_tokenized[:5])
for i in range(5):
  print(i, corpus[i])
corpus1 = []

for review in corpus:
  if isinstance(review,str):
    review = review.split()
    review = [re.sub('[^A-Za-z]+', '', x) for x in review]
    review = [x.lower() for x in review if len(x) > 0]
    corpus1.append(' '.join(review))
# We are using nltk list of stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
             "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
             'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
             'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
             'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
             'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
             'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
             'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to',
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
             'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
             'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
             'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
             "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
             'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
             'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
             'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# Extending stopwords with space
stopwords.append('')

# Converting it to a set
stopwords = set(stopwords)
# Removing stopwords and storing it into a new dict

corpus_sr = [] # Corpus after removing stopwords

for review in corpus1 :
  if isinstance(review, str):
    review = review.split()
    new_review = []
    for x in review:
      if x not in stopwords:
        new_review.append(x)
    corpus_sr.append(" ".join(new_review))
# Creating a list of all words present in review
word_list = []

for review in corpus_sr:
  word_list.extend(review.split())

word_list = list(set(word_list))

print(word_list[:10])
# Stemming
import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

corpus_stemmed = []

for review in corpus_sr:
  review = [porter_stemmer.stem(x) for x in review.split()]
  corpus_stemmed.append(' '.join(review))

for i in range(5):
  print(corpus_sr[i])
  print(corpus_stemmed[i])
# Lemmatization
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

corpus_lemmatized = []

for review in corpus_sr:
  review = [wordnet_lemmatizer.lemmatize(x) for x in review.split()]
  corpus_lemmatized.append(' '.join(review))

for i in range(5):
  print(corpus_sr[i])
  print(corpus_lemmatized[i])
corpus_tokenized = np.array([review.split() for review in corpus_lemmatized])

print(corpus_tokenized[:5])
vocabulary = []
for review in corpus_lemmatized:
  vocabulary.extend(review.split())
vocabulary = list(set(vocabulary))

# Printing some of the first elements of word_list and number of words present in it
print(vocabulary[:5])
print(len(vocabulary))

# Tokenizing
word_to_id = {word:id for id,word in enumerate(vocabulary)}
id_to_word = {id:word for id,word in enumerate(vocabulary)}
m = len(corpus_lemmatized) # m = number of reviews 
n = len(vocabulary) # n = number of unique words

tfm = np.zeros((m, n),dtype=int) # Term frequency matrix
for i in range(m):
  words = corpus_lemmatized[i].split()
  for j in range(len(words)):
    word = words[j]
    tfm[i][word_to_id[word]] += 1 
tmpm = tfm != 0 # Temporary matrix
dft = tmpm.sum(axis = 0) #the number of documents where term t appears
tfidfm = np.multiply(tfm, np.log(m/dft))
U, s, VT = np.linalg.svd(tfm)

K = 2 # number of components

tfm_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), VT[:K, :]))
docs_rep = np.dot(tfm, VT[:K, :].T)
term_rep = np.dot(tfm.T, U[:,:K])
plt.scatter(docs_rep[:,0], docs_rep[:,1])
plt.title("Document Representation")
plt.show()
plt.scatter(term_rep[:,0], term_rep[:,1])
plt.title("Term Representation")
plt.show()
query = 'nice good'


key_word_indices = []

for x in query.split():
  if x in word_to_id.keys():
    key_word_indices.append(word_to_id[x])
key_words_rep = term_rep[key_word_indices,:]     
query_rep = np.sum(key_words_rep, axis = 0)

print (query_rep)
query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

for rank, sort_index in enumerate(query_doc_sort_index):
    print (rank, query_doc_cos_dist[sort_index], corpus[sort_index])
    if rank == 4 : 
      break
query_vector = np.zeros((1,n))
for x in key_word_indices:
  query_vector[0,x] += 1
  
query_vector = np.multiply(query_vector, np.log(m/dft))

query_doc_cos_dist = [cosine(query_vector, tfidfm[i]) for i in range(m)]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

x = []

for rank, sort_index in enumerate(query_doc_sort_index):
    print (rank, query_doc_cos_dist[sort_index], corpus[sort_index])
    x.append(corpus[sort_index])
    if rank == 4 : 
      break