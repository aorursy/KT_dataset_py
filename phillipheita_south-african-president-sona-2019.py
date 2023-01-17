# Web scraping, pickle imports

import requests

from bs4 import BeautifulSoup

import pickle

import pandas as pd



# Scrapes transcript data from https://m.news24.com

def url_to_transcript(url):

    '''Returns transcript data specifically from https://m.news24.com.'''

    page = requests.get(url).text

    soup = BeautifulSoup(page, "lxml")

    text = [p.text for p in soup.find(class_="article_content").find_all('p')]

    print(url)

    return text



# URLs of transcript of Ramaphosa's SONA

url = 'https://m.news24.com/Columnists/GuestColumn/full-text-we-are-committed-to-building-an-ethical-state-says-president-cyril-ramaphosa-in-2019-sona-speech-20190620'



# President name

president = ['Ramaphosa']
# # Actually request transcript

# Scraping doesn't seem to work here, so I import a text file with the speech instead :/

# Someone please help me out here :) 

#sona = url_to_transcript('https://m.news24.com/Columnists/GuestColumn/full-text-we-are-committed-to-building-an-ethical-state-says-president-cyril-ramaphosa-in-2019-sona-speech-20190620')



import os          

os.chdir('/kaggle/')

# You can list all of the objects in a directory by passing the file path to the os.listdir( ) function:

os.listdir('/kaggle/input')

#Read in the text file

text = []

sona = open('input/south-african-sona-2019/SONA.txt','r')

if sona.mode == 'r':

    contents =sona.read()

    text.append(contents)
# Have a quick look at the speech

sona = text.copy()

sona
dict1 = {'Ramaphosa':sona}

# We are going to change this to key: president, value: string format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text

# Combine it!

data_combined = {key: [combine_text(value)] for (key, value) in dict1.items()}

data_combined
# We can either keep it in dictionary format or put it into a pandas dataframe

import pandas as pd

# Increase the column width

pd.set_option('max_colwidth',150)



data_df = pd.DataFrame.from_dict(data_combined).transpose()

data_df.columns = ['transcript']

data_df = data_df.sort_index()

data_df
# Let's take a look at the transcript

data_df.transcript.loc['Ramaphosa']
# Apply a first round of text cleaning techniques

import re

import string



def clean_text_round1(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    text = text.lower()

    # if any of these punctuation marks in (string.punctuation) get rid of it

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # Getting rid of words that contain any numbers in them

    text = re.sub('\w*\d\w*', '', text)

    return text



round1 = lambda x: clean_text_round1(x)
# Let's take a look at the updated text

data_clean = pd.DataFrame(data_df['transcript'].apply(round1))

data_clean
# Apply a second round of cleaning

def clean_text_round2(text):

    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''

    text = re.sub('[‘’“”…]', '', text)

    text = re.sub('\n', '', text)

    text = re.sub('ramaphosa', '', text)

    #text = re.sub('south', '', text)

    return text



round2 = lambda x: clean_text_round2(x)
# Let's take a look at the updated text

data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
df = data_clean.copy()
# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(stop_words='english')

data_cv = cv.fit_transform(df.transcript)

data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

data_dtm.index = df.index

data_dtm
data = data_dtm.transpose()
# Find the top 1000 words said by the president

top_dict = {}

for c in data.columns:

    top = data[c].sort_values(ascending=False)

    top_dict[c]= list(zip(top.index, top.values))
# Look at the most common top words --> add them to the stop word list

from collections import Counter



# Let's first pull out the top 100 words for each comedian

words = []

for president in data.columns:

    top = [word for (word, count) in top_dict[president]]

    for t in top:

        words.append(t)
# Let's update our document-term matrix with the new list of stop words

from sklearn.feature_extraction import text 

from sklearn.feature_extraction.text import CountVectorizer



# Recreate document-term matrix

cv = CountVectorizer(stop_words='english')

data_cv = cv.fit_transform(data_clean.transcript)

data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

data_stop.index = data_clean.index
from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

import matplotlib.pyplot as plt

import nltk

%matplotlib inline





img1 = imread("input/sa-map/SA_map.jpg")

hcmask1 = img1

#FIX: double check whether words_except_stop_dist is necessary 

words_except_stop_dist = nltk.FreqDist(w for w in words if w not in STOPWORDS) 

wordcloud = WordCloud(width = 3000,height = 2000,stopwords=STOPWORDS,background_color='#FBF8EF',mask=hcmask1).generate(" ".join(words_except_stop_dist))

plt.imshow(wordcloud,interpolation = 'bilinear')

fig=plt.gcf()

fig.set_size_inches(10,12)

plt.axis('off')

plt.title("Ramaphosa's SONA 2019",fontsize=22)

plt.tight_layout(pad=0)

plt.show()
# Create lambda functions to find the polarity and subjectivity of the transcript

from textblob import TextBlob



pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



data_clean['polarity'] = data_clean['transcript'].apply(pol)

data_clean['subjectivity'] = data_clean['transcript'].apply(sub)

data_clean
# Split each routine into 10 parts

import numpy as np

import math



def split_text(text, n=20):

    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''



    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text

    length = len(text)

    size = math.floor(length / n)

    start = np.arange(0, length, size)

    

    # Pull out equally sized pieces of text and put it into a list

    split_list = []

    for piece in range(n):

        split_list.append(text[start[piece]:start[piece]+size])

    return split_list
# Let's create a list to hold all of the pieces of text

list_pieces = []

for t in data_clean.transcript:

    split = split_text(t)

    list_pieces.append(split)
# Have a look at the first piece of text

list_pieces[0][0]
# Calculate the polarity for each piece of text

polarity_transcript = []

for lp in list_pieces:

    polarity_piece = []

    for p in lp:

        polarity_piece.append(TextBlob(p).sentiment.polarity)

    polarity_transcript.append(polarity_piece)

    

polarity_transcript
plt.rcParams['figure.figsize'] = [16, 12]



plt.plot(polarity_transcript[0])

plt.plot(np.arange(0,20), np.zeros(20))

plt.ylim(ymin=-.2, ymax=.3)

plt.title('SONA 2019 Speech Sentiment over time',fontsize=12)

plt.show()
# Input for LDA is the document-term matrix

data = data_dtm.copy()
# Import the necessary modules for LDA with gensim

from gensim import matutils, models

import scipy.sparse
# One of the required inputs is a term-document matrix

tdm = data.transpose()

tdm.head(6)
# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus

sparse_counts = scipy.sparse.csr_matrix(tdm)

corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix

id2word = dict((v, k) for k, v in cv.vocabulary_.items())
# we need to specify two other parameters as well - the number of topics and the number of passes

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)

lda.print_topics()
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)

lda.print_topics()
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)

lda.print_topics()
# Let's create a function to pull out nouns from a string of text

from nltk import word_tokenize, pos_tag



def nouns(text):

    '''Given a string of text, tokenize the text and pull out only the nouns.'''

    is_noun = lambda pos: pos[:2] == 'NN'

    tokenized = word_tokenize(text)

    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 

    return ' '.join(all_nouns)
data_clean
# Apply the nouns function to the transcripts to filter only on nouns

data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))

data_nouns
# Create a new document-term matrix using only nouns

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer



# Re-add the additional stop words since we are recreating the document-term matrix

add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',

                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)



# Recreate a document-term matrix with only nouns

cvn = CountVectorizer(stop_words=stop_words)

data_cvn = cvn.fit_transform(data_nouns.transcript)

data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())

data_dtmn.index = data_nouns.index

data_dtmn
# Create the gensim corpus

corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))



# Create the vocabulary dictionary

id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())
# Let's start with 2 topics

ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)

ldan.print_topics()
ldan = models.LdaModel(corpus=corpusn, num_topics=3, id2word=id2wordn, passes=10)

ldan.print_topics()
ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=10)

ldan.print_topics()
# Let's create a function to pull out nouns from a string of text

def nouns_adj(text):

    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''

    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'

    tokenized = word_tokenize(text)

    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 

    return ' '.join(nouns_adj)
# Apply the nouns function to the transcripts to filter only on nouns

data_nouns_adj = pd.DataFrame(data_clean.transcript.apply(nouns_adj))

data_nouns_adj
# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df

cvna = CountVectorizer(stop_words=stop_words)

data_cvna = cvna.fit_transform(data_nouns_adj.transcript)

data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())

data_dtmna.index = data_nouns_adj.index

data_dtmna
# Create the gensim corpus

corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))



# Create the vocabulary dictionary

id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())
ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=10)

ldana.print_topics()
ldana = models.LdaModel(corpus=corpusna, num_topics=3, id2word=id2wordna, passes=10)

ldana.print_topics()
ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=10)

ldana.print_topics()
# Our final LDA model (for now)

ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=80)

ldana.print_topics()