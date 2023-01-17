# Code cell just to show what all data is there

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import numpy

# import gensim

import nltk

# import seaborn

import pandas as pd
summary = pd.read_csv('/kaggle/input/news-summary/news_summary.csv', encoding='iso-8859-1')
raw = pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv', encoding='iso-8859-1')
raw.head() #how does raw DataFrame look like
raw.iloc[0,0], raw.iloc[0,1] #viewing the contents inside raw's 0th element
#making a word_count column in the DataFrame

raw['word_count'] = raw['text'].apply(lambda x: len(str(x).split(" ")))

raw.head()
raw.word_count.describe() #describes what all data is there and other stats
freq = pd.Series(' '.join(raw['text']).split()).value_counts()[:20]

freq
freq.plot.barh();
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer
# Stopwords are all the commonly used english words which don't contribute to keywords such as 'as', 'are' etc

stop_words = set(stopwords.words("english"))

# Creating a customized stopword list from data shown below after several iterations

new_words = ["using", "show", "result", "large", "also", "iv",

             "one", "two", "new", "previously", "shown", "year", "old", "said", "reportedly",

             "added", "u", "day", "time"]

stop_words = stop_words.union(new_words) #customised stopwords added to previous stopword
# Creating a new list of texts called corpus where the following things are removed

corpus = []

for i in range(0, 3847):

    # Remove punctuations

    text = re.sub('[^a-zA-Z]', ' ', raw['text'][i])

    # Convert to lowercase

    text = text.lower()

    # Remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # Remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    # Convert to list from string

    text = text.split()

    # Stemming

    ps=PorterStemmer()

    # Lemmatisation

    lem = WordNetLemmatizer()

    text = [lem.lemmatize(word) for word in text if not word in  stop_words] 

    text = " ".join(text)

    corpus.append(text)
# The original text

raw.iloc[0,1]
# After removing stopwords, punctions and normalizing to root words

corpus[0]
# Word cloud which is pretty good for representation

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# This is just to represent such representations are also possible

# Here representation only done for the first text

num_text = 1

for i in range(num_text):

    wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=100,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(corpus[i]))

    fig = plt.figure(1)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    fig.savefig("word1.png", dpi=900)
# Tells us the max keywords used without including stopwords in the whole corpus and we add such words to new stop words

freq = pd.Series(' '.join(corpus).split()).value_counts()[:20]

freq 
# Corpus cell number chosen(arbritarily)

corpn = 100
# Tokenizes and builds a vocabulary

from sklearn.feature_extraction.text import CountVectorizer

import re

cv=CountVectorizer(stop_words=stop_words, max_features=10, ngram_range=(1,3))
#this can be put inside a loop to get key words for all articles

corpi = [corpus[corpn]] #changing the number here will give us the key words for that specific article

X=cv.fit_transform(corpi)

list(cv.vocabulary_.keys())[:10]
corpi
realtext = raw.iloc[corpn,1]

realtext
from sklearn.feature_extraction.text import TfidfTransformer

 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(X)

# get feature names

feature_names=cv.get_feature_names()

 

# fetch document for which keywords needs to be extracted

doc=corpus[corpn]

 

#generate tf-idf for the given document

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#tf_idf sorting in descending order

from scipy.sparse import coo_matrix

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

 

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results

#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10

keywords=extract_topn_from_vector(feature_names,sorted_items,10)

 

# now print the results

print("\nAbstract:")

print(doc)

print("\nKeywords:")

for k in keywords:

    print(k,keywords[k])