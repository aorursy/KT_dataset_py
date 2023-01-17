
!pip install mglearn
import re
import numpy as np
import nltk
from nltk.tag import UnigramTagger
from nltk.corpus import words, stopwords
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from collections import Counter
import requests
import os
import pandas as pd
import bs4
import sys
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis.sklearn

from html.parser import HTMLParser
import spacy
from itertools import combinations 
import warnings
warnings.filterwarnings('ignore')
from operator import itemgetter
from nltk.corpus import wordnet as wn
from html.parser import HTMLParser
import pprint
import string
import statistics
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import time
import mglearn
from wordcloud import WordCloud
print("Below, I am providing system information.")
print("=========================================")
print("Sklearn version -",sklearn.__version__)
print("NLTK version -",nltk.__version__)
print("SpaCy version -",spacy.__version__)
print("BeautifulSoup version -",bs4.__version__)
print("Numpy version -",np.__version__)
print("Pandas version -",pd.__version__)
print("Python and Anaconda version -",sys.version)
stopwords = nltk.corpus.stopwords.words('english')

# page 268 adding more words into the original list
stopwords = stopwords + ['mr', 'mrs', 'family','come', 'go', 'get', 'tell', 'listen', 'one', 'two', 'three', 'four', 
                         'five', 'six', 'seven', 'eight', 'nine', 'zero', 'join', 'find', 'make', 'say', 
                         'ask', 'tell', 'see', 'try', 'back', 'also','movie',
                         '1','2','3','4','5','6','7','8','9','10','0',
                         'film', 'movie', 'watch', 'cinema', 'scene','action', 'fighting','story', '3D'
                         'show', 'get','tell', 'listen']

# github of dipanjanS [2]
CONTRACTION_MAP = {"ain't": "is not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",
                   "could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not",
                   "doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
                   "haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
                   "he'll've": "he he will have","he's": "he is","how'd": "how did","how'd'y": "how do you",
                   "how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have","I'll": "I will",
                   "I'll've": "I will have","I'm": "I am","I've": "I have","i'd": "i would","i'd've": "i would have",
                   "i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have","isn't": "is not",
                   "it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is",
                   "let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
                   "mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
                   "needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                   "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have",
                   "she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have",
                   "she's": "she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have",
                   "so've": "so have","so's": "so as","that'd": "that would","that'd've": "that would have","that's": "that is",
                   "there'd": "there would","there'd've": "there would have","there's": "there is","they'd": "they would",
                   "they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are",
                   "they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would","we'd've": "we would have",
                   "we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not",
                   "what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is",
                   "what've": "what have","when's": "when is","when've": "when have","where'd": "where did",
                   "where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have",
                   "who's": "who is","who've": "who have","why's": "why is","why've": "why have","will've": "will have",
                   "won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
                   "wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
                   "you'd've": "you would have","you'll": "you will","you'll've": "you will have","you're": "you are",
                   "you've": "you have"}

# Ben Brock's Analyzing Movie Reviews - Sentiment Analysis notebook
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

# page 175 to tokenize the text
def tokenize(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

# page 118 to expand the contractions
def expand_contractions(sentence, CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match)\
                                if CONTRACTION_MAP.get(match)\
                                else CONTRACTION_MAP.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

# page 176 remove the special symbols and characters
def remove_special_characters(text):
    tokens = tokenize(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# page 120 to remove the stop words
def remove_stopwords(sentence):
    tokens = nltk.word_tokenize(sentence)
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd 

review_dataset = pd.read_csv('/kaggle/input/1943polymerase/1943tibbleABSTRACT_polymeraseTitleABSGroup.csv')
print('Let us take a brief look at the data:\n',review_dataset.head())

review = np.array(review_dataset['ABS'])

# normalizing the train data
print('Normalize dataset needs:')
html_parser = HTMLParser()
normed_review = []
start = time.time()
for text in review:
    text = html_parser.unescape(text)
    text = expand_contractions(text, CONTRACTION_MAP)
    text = text.lower()
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    normed_review.append(text)
normed_review = [str (item) for item in normed_review]
normed_review = [item for item in normed_review if not isinstance(item, int)]

end = time.time()
print(end - start,'seconds')
print(normed_review)
# page 345 to extract features
def build_feature_matrix(documents, feature_type='frequency', ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf': 
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    else: 
        raise Exception("Wrong feature type entered. Possible values:'binary', 'frequency', 'tfidf'")
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    return vectorizer, feature_matrix

vectorizer, feature_matrix = build_feature_matrix(normed_review, feature_type='frequency',min_df=0.01, max_df=0.55)
print("The dimension of the feature matrix is",feature_matrix.shape)
feature_names = vectorizer.get_feature_names()
# lets set a base number of components
n_comp = 6
start = time.time()
# LDA set up
lda = LatentDirichletAllocation(n_components = n_comp,
                                random_state= 2019,
                                learning_method= 'online',
                                verbose = True)


data_lda_p = lda.fit_transform(feature_matrix)
end = time.time()
print(end - start,'seconds')
# print out the topics using a for loop for reviews
for idx, topic in enumerate(lda.components_):
    print("Topic",idx+1)
    print([(vectorizer.get_feature_names()[i], topic[i])
           for i in topic.argsort()[:-8 - 1:-1]])
pyLDAvis.enable_notebook()

dashboard = pyLDAvis.sklearn.prepare(lda, feature_matrix, vectorizer)

dashboard
review_wordcloud = ' '.join(normed_review)

wordcloud = WordCloud().generate(review_wordcloud)
plt.figure(figsize = (16, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# more components
n_comp = 10
start = time.time()
# LDA set up
lda = LatentDirichletAllocation(n_components = n_comp,
                                random_state= 2019,
                                learning_method= 'online',
                                verbose = True)


data_lda_p = lda.fit_transform(feature_matrix)
end = time.time()
print(end - start,'seconds')
# print out the topics using a for loop for reviews
for idx, topic in enumerate(lda.components_):
    print("Topic",idx+1)
    print([(vectorizer.get_feature_names()[i], topic[i])
           for i in topic.argsort()[:-10 - 1:-1]])
pyLDAvis.enable_notebook()

dashboard = pyLDAvis.sklearn.prepare(lda, feature_matrix, vectorizer)

dashboard
## 6 groups seem to have better separations among groups