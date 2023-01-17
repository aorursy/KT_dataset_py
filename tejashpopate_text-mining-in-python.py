#nltk.download()

!pip install textmining3

!pip install vaderSentiment
# import required 

import pandas as pd

import csv

from textblob import TextBlob

#from textblob.sentiments import NaiveBayesAnalyzers

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

import string

import nltk

import textmining

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS

# set the working dierectory

import os

os.chdir("../input")
# load the dataset

post=pd.read_csv("Post.csv")

post.head(20)
# select few text

post=post.iloc[:1000,]

post.shape
# define the stopwords and also import predefined ones

stop=set(stopwords.words("english"))

# Extract punctuation marks

punct_exclude=set(string.punctuation)
# perform preoprocessing using function below

def clean(doc):

    stop_free=" ".join([i for i in doc.lower().split() if i not in stop])

    punct_free=''.join(ch for ch in stop_free if ch not in punct_exclude)

    num_free=''.join(i for i in punct_free if not i.isdigit())

    return num_free



post_corpus=[clean(post.iloc[i,1]) for i in range(0,post.shape[0])]
print(type(post_corpus))

print(post_corpus[14])
# create term document matrix

tdm=textmining.TermDocumentMatrix() # use a function from textmining library

for i in post_corpus:

    #print(i)

    tdm.add_doc(i)# update the matrix with each variable conversion
type(tdm)
os.chdir("../working")


#write tdm into dataframe

tdm.write_csv("TDM_DataFrame.csv",cutoff= 1) #cutoff won't consider 1st line
def buildMatrix(self,document_list):

        print("building matrix...")

        tdm = textmining.TermDocumentMatrix()

        for doc in document_list:

             tdm.add_doc(doc)

        #write tdm into dataframe

        tdm.write_csv(r'path\matrix.csv', cutoff=1)
df=pd.read_csv("TDM_DataFrame.csv")
df.head(20)
post.shape
df.shape
#plot wordcloud

wordcloud= WordCloud(width=1000,height=500, stopwords=STOPWORDS, background_color='white').generate(''.join(post['Post']))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()

# Sentiment Analysis TextBlob implementation

# create an empty dataframe to store the results

FinalResult=pd.DataFrame()

for i in range(0,post.shape[0]):

    #print(i)

    blob=TextBlob(post.iloc[i,1])

    temp=pd.DataFrame({"Comments":post.iloc[i,1],"Polarity":blob.sentiment.polarity},index=[0])

    FinalResult=FinalResult.append(temp)
print(FinalResult.shape)

FinalResult.head()
FinalResult.describe()
# Sentiment Analysis using "VADER"

# create an empty dataframe to store the results

FinalResult_vader=pd.DataFrame()

# initialize the engine

analyzer=SentimentIntensityAnalyzer()

for i in range(0,post.shape[0]):

    #print(i)

    snt=analyzer.polarity_scores(post.iloc[i,1]) # gives the output in dictionary form shown below

    temp=pd.DataFrame({"Comments":post.iloc[i,1],"Polarity":snt.items()},index=[0])

    FinalResult_vader=FinalResult_vader.append(temp)

snt
FinalResult_vader.head()