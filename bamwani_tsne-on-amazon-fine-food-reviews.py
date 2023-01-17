%matplotlib inline 



import warnings

warnings.filterwarnings("ignore")





import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os



cwd= os.getcwd() #to get the current working directory location

print (cwd)
# using the SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite') 

# filtering only positive and negative reviews i.e. not taking into consideration those reviews with Score=3

# SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000, will give top 500000 data points

# you can change the number to any other number based on your computing power

# filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 500000""", con) 



filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 LIMIT 5000""", con) 



# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.

def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



# changing reviews with score less than 3 to be negative and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative

print("Number of data points in our data", filtered_data.shape)

filtered_data.head(3)
display = pd.read_sql_query(""" 

SELECT UserId, ProductId, ProfileName, Time, Score, Text, COUNT(*)

FROM Reviews

GROUP BY UserId

HAVING COUNT(*)>1

""", con)
print(display.shape)

display.head(5)
# checking a random review.

display[display['UserId']=='AZY10LLTJ71NX']
display['COUNT(*)'].sum()
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display.head()
#Sorting data according to ProductId in ascending order

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductID

""", con)



display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

final['Score'].value_counts()
# printing some random reviews

sent_0 = final['Text'].values[0]

print(sent_0)

print("="*50)



sent_100 = final['Text'].values[100]

print(sent_100)

print("="*50)



sent_300 = final['Text'].values[300]

print(sent_300)

print("="*50)



sent_900 = final['Text'].values[900]

print(sent_900)

print("="*50)
# remove urls from text python: https://stackoverflow.com/a/40823105/4084039

sent_0 = re.sub(r"http\S+", "", sent_0)

sent_100 = re.sub(r"http\S+", "", sent_100)

sent_300 = re.sub(r"http\S+", "", sent_300)

sent_900 = re.sub(r"http\S+", "", sent_900)



print(sent_0)
# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element

from bs4 import BeautifulSoup



soup = BeautifulSoup(sent_0, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_100, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_300, 'lxml')

text = soup.get_text()

print(text)

print("="*50)



soup = BeautifulSoup(sent_900, 'lxml')

text = soup.get_text()

print(text)
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
sent_300 = decontracted(sent_300)

print(sent_300)

print("="*50)
#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039

sent_0 = re.sub("\S*\d\S*", "", sent_0).strip()

print(sent_0)
#remove spacial character: https://stackoverflow.com/a/5843547/4084039

sent_300 = re.sub('[^A-Za-z0-9]+', ' ', sent_300)

print(sent_300)
# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

# <br /><br /> ==> after the above steps, we are getting "br br"

# we are including them into stop words list

# instead of <br /> if we have <br/> these tags would have revmoved in the 1st step



stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
from bs4 import BeautifulSoup

from tqdm import tqdm

preprocessed_reviews = []

# tqdm is for printing the status bar

for sentance in tqdm(final['Text'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentance.strip())
preprocessed_reviews[500]
# https://stackoverflow.com/q/23748995

score=final['Score'] #storing all scores in a new series.

print(type(score))

print(score.shape)
#BoW

count_vect = CountVectorizer() #in scikit-learn

count_vect.fit(preprocessed_reviews)

print("some feature names ", count_vect.get_feature_names()[:10])

print('='*50)



final_counts = count_vect.transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_counts))

print("the shape of out text BOW vectorizer ",final_counts.get_shape())

print("the number of unique words ", final_counts.get_shape()[1])
#bi-gram, tri-gram and n-gram



#removing stop words like "not" should be avoided before building n-grams

# count_vect = CountVectorizer(ngram_range=(1,2))

# please do read the CountVectorizer documentation http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# you can choose these numebrs min_df=10, max_features=5000, of your choice

count_vect = CountVectorizer(ngram_range=(1,2))

final_bigram_counts = count_vect.fit_transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_bigram_counts))

print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(preprocessed_reviews)

print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])

print('='*50)



final_tf_idf = tf_idf_vect.transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_tf_idf))

print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])
# Train your own Word2Vec model using your own text corpus

i=0

list_of_sentance=[]

for sentance in preprocessed_reviews:

    list_of_sentance.append(sentance.split())
w2v_model=Word2Vec(list_of_sentance,min_count=5,size=50, workers=4)

print(w2v_model.wv.most_similar('great'))

print('='*50)

print(w2v_model.wv.most_similar('worst'))
w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum 5 times ",len(w2v_words))

print("sample words ", w2v_words[0:30])
# average Word2Vec

# compute average word2vec for each review.

sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors.append(sent_vec)

print(len(sent_vectors))

print(len(sent_vectors[0]))
# S = ["abc def pqr", "def def def abc", "pqr pqr def"]

model = TfidfVectorizer()

model.fit(preprocessed_reviews)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec

tfidf_feat = model.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list

row=0;

for sent in tqdm(list_of_sentance): # for each review/sentence 

    sent_vec = np.zeros(50) # as word vectors are of zero length

    weight_sum =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

#             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_vectors.append(sent_vec)

    row += 1
# As BoW and TF-IDF gives sparse matrix, and TSNE only accepts Dense matrix.so we need to convert sparse matrix to dense

from sklearn.preprocessing import StandardScaler

print(final_bigram_counts.shape)

std_data = StandardScaler(with_mean = False).fit_transform(final_bigram_counts)

print(std_data.shape)

type(std_data)

std_data=std_data.todense()

print(type(std_data))
from sklearn.manifold import TSNE

from sklearn import datasets

import pandas as pd

import matplotlib.pyplot as plt



model = TSNE(n_components=2, perplexity=45, learning_rate=100, n_iter = 500, random_state=0)



for_tsne = model.fit_transform(std_data)

# if x is a sparse matrix you need to pass it as X_embedding = tsne.fit_transform(x.todense()) , .toarray() will convert the sparse matrix into dense matrix



for_tsne = np.vstack((for_tsne.T, score)).T

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

sns.FacetGrid(for_tsne_df, hue="Score", size=10).map(plt.scatter, 'Dimension_x', 'Dimension_y').add_legend()

plt.title("TSNE for Bag of Words")
from sklearn.preprocessing import StandardScaler



std_data1 = StandardScaler(with_mean = False).fit_transform(final_tf_idf)

print(std_data1.shape)

type(std_data1)

std_data1=std_data1.todense()

type(std_data1)


model = TSNE(n_components=2, perplexity=40, learning_rate=100, n_iter=500)



for_tsne = model.fit_transform(std_data1)

# if x is a sparse matrix you need to pass it as X_embedding = tsne.fit_transform(x.todense()) , .toarray() will convert the sparse matrix into dense matrix



for_tsne = np.vstack((for_tsne.T, score)).T

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

sns.FacetGrid(for_tsne_df, hue="Score", size=6).map(plt.scatter, 'Dimension_x', 'Dimension_y').add_legend()

plt.title("TF-IDF")
from sklearn.preprocessing import StandardScaler



std_data2 = StandardScaler(with_mean = False).fit_transform(sent_vectors)

print(std_data2.shape)

type(std_data2)

type(std_data2)


model = TSNE(n_components=2, perplexity=30, learning_rate=600, n_iter=500)



for_tsne = model.fit_transform(std_data2)

# if x is a sparse matrix you need to pass it as X_embedding = tsne.fit_transform(x.todense()) , .todense() will convert the sparse matrix into dense matrix



for_tsne = np.vstack((for_tsne.T, score)).T

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

sns.FacetGrid(for_tsne_df, hue="Score", size=6).map(plt.scatter, 'Dimension_x', 'Dimension_y').add_legend()

plt.title("AVG W2V")
from sklearn.preprocessing import StandardScaler



std_data3 = StandardScaler(with_mean = False).fit_transform(tfidf_sent_vectors)

print(std_data3.shape)

type(std_data3)

type(std_data3)



model = TSNE(n_components=2, perplexity=40, learning_rate=600, n_iter=500)



for_tsne = model.fit_transform(std_data3)

# if x is a sparse matrix you need to pass it as X_embedding = tsne.fit_transform(x.todense()) , .todense() will convert the sparse matrix into dense matrix



for_tsne = np.vstack((for_tsne.T, score)).T

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

sns.FacetGrid(for_tsne_df, hue="Score", size=6).map(plt.scatter, 'Dimension_x', 'Dimension_y').add_legend()

plt.title("Weighted W2V")
# store final table into an SQlLite table for future.

conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)