import os

import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



from gensim import corpora, models, similarities



import warnings

warnings.filterwarnings("ignore")

import nltk

import re

import pandas as pd
datafile = '../input/detik18/detik18.csv'
tweets = pd.read_csv(datafile, encoding='latin1')

tweets = tweets.assign(Time=pd.to_datetime(tweets.Time))



tweets.head(10)
range(len(tweets['Tweet']))
'''import plotly.plotly as py

import plotly.graph_objs as go

'''

tweets['Time'] = pd.to_datetime(tweets['Time'], format='%y-%m-%d %H:%M:%S')

tweetsT = tweets['Time']



trace = go.Histogram(

    x=tweetsT,

    marker=dict(

        color='blue'

    ),

    opacity=0.75

)



layout = go.Layout(

    title='Tweet Activity Over Years',

    height=450,

    width=1200,

    xaxis=dict(

        title='Month and year'

    ),

    yaxis=dict(

        title='Tweet Quantity'

    ),

    bargap=0.2,

)



data = [trace]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
#initialize stopWords

stopWords = []



#start replaceTwoOrMore

def replaceTwoOrMore(s):

    #look for 2 or more repetitions of character and replace with the character itself

    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)

    return pattern.sub(r"\1\1", s)

#end
#start getStopWordList

def getStopWordList(stopWordListFileName):

    #read the stopwords file and build a list

    stopWords = []

    stopWords.append('AT_USER')

    stopWords.append('URL')



    fp = open(stopWordListFileName, 'r')

    line = fp.readline()

    while line:

        word = line.strip()

        stopWords.append(word)

        line = fp.readline()

    fp.close()

    return stopWords

#end
#start getStopWordList

def getStopWordList(stopWordListFileName):

    #read the stopwords file and build a list

    stopWords = []

    stopWords.append('AT_USER')

    stopWords.append('URL')



    fp = open(stopWordListFileName, 'r')

    line = fp.readline()

    while line:

        word = line.strip()

        stopWords.append(word)

        line = fp.readline()

    fp.close()

    return stopWords

#end
#import regex

import re

#start process_tweet

def processTweet(tweet):

    # process the tweets

    #Convert to lower case

    tweet = tweet.lower()

    #Convert www.* or https?://* to URL

    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)

    #Convert @username to AT_USER

    tweet = re.sub('@[^\s]+','AT_USER',tweet)

    #Remove additional white spaces

    tweet = re.sub('[\s]+', ' ', tweet)

    #Replace #word with word

    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    #trim

    tweet = tweet.strip('\'"')

    return tweet

#end



#start getfeatureVector

def getFeatureVector(tweet):

    featureVector = []

    #split tweet into words

    words = tweet.split()

    for w in words:

        #replace two or more with two occurrences

        w = replaceTwoOrMore(w)

        #strip punctuation

        w = w.strip('\'"?,.')

        #check if the word stats with an alphabet

        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)

        #ignore if it is a stop word

        if(w in stopWords or val is None):

            continue

        else:

            featureVector.append(w.lower())

    return featureVector

#end
#Read the tweets one by one and process it

fp = open('../input/detik18/detik18.csv', 'r')

line = fp.readline()



stopWords = getStopWordList('../input/stopword/stopwordsID.txt')

kalimat = []

while line:

    processedTweet = processTweet(line)

    featureVector = getFeatureVector(processedTweet)

    #print (featureVector)

    kalimat.append(featureVector)

    line = fp.readline()

#end loop

fp.close()

a = kalimat[1:]

corpus=[]

a=[]

for i in range(len(kalimat)):

        a=kalimat[i]

        for t in range (len(a)):

            b=a[t]

            corpus.append(b)

#print(corpus)
corpus[0:5]
import gensim

import logging

import tempfile



TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.corpus import stopwords

from string import punctuation



# remove common words and tokenize

list1 = ['RT','rt']

stoplist = stopwords.words('indonesian') + list(punctuation) + list1



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]
dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'elon.dict'))  # store the dictionary, for future reference

#print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'elon.mm'), corpus)  # store to disk, for later use
from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
total_topics = 5
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#Show first n important word in the topics:

lda.show_topics(total_topics,5)
from collections import OrderedDict



data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}

#data_lda
import pandas as pd



df_lda = pd.DataFrame(data_lda)

print(df_lda.shape)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
df_lda
import pyLDAvis.gensim



pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel