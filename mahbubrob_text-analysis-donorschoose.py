import os
import re # remove text 
import numpy as np 
import pandas as pd 
import time

print('Current Directory: ', os.getcwd())
print('List all files and folders: ', os.listdir())
print('Files inside input directory: ')
print(os.listdir("../input"))
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

%%time
projects = pd.read_csv('../input/Projects.csv', parse_dates = ["Project Posted Date","Project Fully Funded Date"])
projects.tail(1)
projects.shape
print(projects.describe())
projects['Project Cost'].describe().apply(lambda x: format(x, 'f'))
# project essay column, zero-th row
project_essay = projects['Project Essay'][0]
project_essay[0:250]
import nltk
from nltk.tokenize import sent_tokenize

project_essay = projects['Project Essay'][0]
project_essay = sent_tokenize(project_essay)
project_essay[0:5]
import re # remove text 
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay[0:300]
project_essay = project_essay.lower()
project_essay[0:300]
import nltk
from nltk.corpus import stopwords 
import time

projects['Project Essay'][0]

start = time.time()
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()
project_essay = project_essay.split()
end = time.time()

print('Time taken:', end-start)
print('Total Number of Words: ', len(project_essay)) # Total Number of words: 414
print('First 10 words in the list')
project_essay[0:10]
projects['Project Essay'][0]
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()
project_essay = project_essay.split()

project_essay = [[x,project_essay.count(x)] for x in set(project_essay)]

print('Total Number of words: ', len(project_essay)) # 195
project_essay[0:10]
import nltk
from nltk.corpus import stopwords 

projects['Project Essay'][0]
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()
project_essay = project_essay.split()
#[[x,project_essay.count(x)] for x in set(project_essay)]
stop_words = set(stopwords.words('english')) # Use english language and make it faster using set function
project_essay = [word for word in project_essay if not word in stop_words]
print('Total Number of words after applying stopwords: ', len(project_essay)) # 206
project_essay[0:10]
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

projects['Project Essay'][0]
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()

project_essay = project_essay.split()

# Stem
ps = PorterStemmer()

stop_words = set(stopwords.words('english')) 
project_essay = [ps.stem(word) for word in project_essay if not word in stop_words]
print('Total Number of words after applying stopwords: ', len(project_essay))
project_essay[0:10]
import nltk
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer

projects['Project Essay'][0]
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()

project_essay = project_essay.split()

# Stem
sno = SnowballStemmer('english')

stop_words = set(stopwords.words('english')) 
project_essay = [sno.stem(word) for word in project_essay if not word in stop_words]
print('Total Number of words after applying stopwords: ', len(project_essay))
project_essay[0:10]
import nltk
import re # remove text 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

projects['Project Essay'][0]
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()

project_essay = project_essay.split()

# Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
snowball_stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english')) 

project_essay = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in project_essay if not word in stop_words]
print('Total Number of words after applying lemmatizer: ', len(project_essay))
print('First 10 Lemmatizers')
project_essay[0:10]
%%time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

project_essay = projects['Project Essay'][0] # Only one essay
project_essay = sent_tokenize(project_essay)

# Stem
#sno = SnowballStemmer('english')

stop_words = set(stopwords.words('english')) 
project_essay = [word for word in project_essay if not word in stop_words]

vectorizer = CountVectorizer()

bag_of_words = vectorizer.fit_transform(project_essay)
#bag_of_words = vectorizer.transform(project_essay)
#X[0:20]

#project_essay[0:5]
print('Bag of Words(Show 10):')
print(bag_of_words[10])
print('Bully is the word number: ', vectorizer.vocabulary_.get('bully'), 'th, including white space?')
import nltk
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer

projects['Project Essay'][0]
project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][0])
project_essay = project_essay.lower()

project_essay = project_essay.split()

# Stem
sno = SnowballStemmer('english')

stop_words = set(stopwords.words('english')) 
project_essay = [sno.stem(word) for word in project_essay if not word in stop_words]

project_essay = ' '.join(project_essay)

print('Total Number of words after joining: ', len(project_essay))
project_essay[0:250]
projects['Project Essay'].isnull().sum().sum()
projects['Project Essay'].fillna(value='is null', inplace=True)
projects['Project Essay'].isnull().sum().sum()
%%time
import nltk
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# While doing Bag of Words or Tf-idf aka vectorization you can not have nan values. Nan values will create errors in the run time.
projects['Project Essay'].fillna(value='is null', inplace=True)

corpus = []
for i in range(0, 100):
    project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][i])
    project_essay = project_essay.lower()
    project_essay = project_essay.split()

    sns = SnowballStemmer('english')
    stop_words = set(stopwords.words('english')) 

    project_essay = [sns.stem(word) for word in project_essay if not word in stop_words]
    project_essay = ' '.join(project_essay)
    corpus.append(project_essay)

# Doing stowords twice to check    
stop_words = set(stopwords.words('english'))
stop_words.update(('donotremoveessaydivid', 'student', 'help', 'learn', 'need', 
                   'school', 'classroom', 'also', 'class', 'make', 'mani', 'read', 'use', 'work', 
                  'abl', 'come', 'day', 'love', 'project', 'provid', 'skill', 'teach', 'time', 'year',
                  'allow', 'becom', 'children', 'educ', 'get', 'give', 'grade', 'high', 'like', 'live', 
                   'materi', 'new', 'one', 'opportun', 'see', 'teacher', 'want', 'way', 'would'))

print('Corpus Length: ', len(corpus))    

bag_of_words_vectorizer = CountVectorizer(stop_words=stop_words, 
                               analyzer='word', 
                               ngram_range=(1, 3), 
                               max_df=1.0, 
                               min_df=1, # value 1 works and shows n-gram for large features and corpus, value 1.0 wants lower min df
                               max_features = 100)

# Transforms the data into a bag of words
bag_of_words_train = bag_of_words_vectorizer.fit(corpus)
bag_of_words = bag_of_words_vectorizer.transform(corpus)

#X = bag_of_words.fit_transform(corpus)

print('Bag of Words:')
print('(Doc, Word)   Freq')
print(bag_of_words)

# Print the first 10 features of the count_vec https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments
print("Every feature:\n{}".format(bag_of_words_vectorizer.get_feature_names()[:10]))
print("\nEvery 3rd feature:\n{}".format(bag_of_words_vectorizer.get_feature_names()[::3]))
%%time
import nltk
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

projects['Project Essay'].fillna(value='Missing', inplace=True)

corpus = []
sns = SnowballStemmer('english')
for i in range(0, 1000):
    project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][i])
    project_essay = project_essay.lower()
    project_essay = project_essay.split()

    
    project_essay = [sns.stem(word) for word in project_essay]
    project_essay = ' '.join(project_essay)
    corpus.append(project_essay)

# Doing stowords twice to check    
stop_words = set(stopwords.words('english'))
stop_words.update(('donotremoveessaydivid', 'student', 'help', 'learn', 'need', 
                   'school', 'classroom', 'also', 'class', 'make', 'mani', 'read', 'use', 'work', 
                  'abl', 'come', 'day', 'love', 'project', 'provid', 'skill', 'teach', 'time', 'year',
                  'allow', 'becom', 'children', 'educ', 'get', 'give', 'grade', 'high', 'like', 'live', 
                   'materi', 'new', 'one', 'opportun', 'see', 'teacher', 'want', 'way', 'would'))

tfidf_vectorizer = TfidfVectorizer(
                               stop_words=stop_words, 
                               analyzer='word', 
                               ngram_range=(1, 4), 
                               max_df=1.0, 
                               min_df=1, # value 1 works and shows n-gram for large features and corpus, value 1.0 wants lower min df
                               sublinear_tf=True, # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
                               max_features=1000)

start_time = time.time()
tfidf_train = tfidf_vectorizer.fit(corpus)
tfidf =  tfidf_vectorizer.transform(corpus)
end_time = time.time()

print('TFIDF:')
print('(Doc, Word)   Importance')
print(tfidf[:10])

print("10 features:\n{}".format(tfidf_vectorizer.get_feature_names()[:10]))
print("\nEvery 3rd feature:\n{}".format(tfidf_vectorizer.get_feature_names()[:100:3]))

print('Time to train vectorizer and transform training text: %0.2fs' % (end_time - start_time))

import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import corpora, models, similarities

# Gensim specific logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# >>> dataset = api.load("text8")
# >>> dct = Dictionary(dataset)  # fit dictionary
# >>> corpus = [dct.doc2bow(line) for line in dataset]  # convert dataset to BoW format
# >>>
# >>> model = TfidfModel(corpus)  # fit model
# >>> vector = model[corpus[0]]  # apply model

projects['Project Essay'].fillna(value='Missing', inplace=True)


corpus = []
sns = SnowballStemmer('english')
for i in range(0, 1000):
    project_essay = re.sub('[^a-zA-Z]', ' ', projects['Project Essay'][i])
    project_essay = [sns.stem(word) for word in project_essay]
    corpus.append(project_essay)

dictionary = corpora.Dictionary(corpus)
corpus_bow_format = [dictionary.doc2bow(text) for text in corpus]

# fit model, gensim tfidf requires corpus to be in a Bag of Words format 
#https://radimrehurek.com/gensim/tut2.html
model = models.TfidfModel(corpus_bow_format) 

tfidf = model[corpus_bow_format[0]] # apply model

print(tfidf)
%%time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

projects['Project Essay'].fillna(value='Missing', inplace=True)

# Doing stowords twice to check    
stop_words = set(stopwords.words('english'))
stop_words.update(('donotremoveessaydivid'))

project_essay = projects['Project Essay'].tolist()

tfidf_vectorizer = TfidfVectorizer(input = project_essay,
                                   stop_words = stop_words, 
                                   max_features = 1000
                                  )
start_time = time.time()
tfidf_train = tfidf_vectorizer.fit(project_essay)
tfidf_matrix =  tfidf_vectorizer.transform(project_essay)
end_time = time.time()

print(tfidf_matrix.shape)
print('Time to train vectorizer and transform training text: %0.2fs' % (end_time - start_time)) # will take around 7-8 mins
%%time
### Get Similarity using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
#similarity_using_unigram = cosine_similarity(tfidf_matrix)
import spacy

nlp = spacy.load('en')

# From spaCy website
# Dense, real valued vectors representing distributional similarity information 
# are now a cornerstone of practical NLP. The most common way to train these vectors 
# is the word2vec family of algorithms. If you need to train a word2vec model, 
# we recommend the implementation in the Python library Gensim.
