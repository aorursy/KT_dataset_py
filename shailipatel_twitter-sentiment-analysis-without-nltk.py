#import essential libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/twitter-sentiment-anaylsis/train_2kmZucJ.csv")

test = pd.read_csv("../input/twitter-sentiment-anaylsis/test_oJQbWVk.csv")
train.head()
train.tweet[:20]
badword='$&@*#'   #we initialize a variable which contain this profane term

tt=train['tweet'].apply(lambda x:x.split()) #tokenize the tweet text
tt[:5]
z=[]   #create a list 

for i in tt: #run through the loop 

    l=[]

    for j in i:

        if j.__contains__('$&@*#'):     #and compare each and every word to badword

            j=j.replace(j,'bad')        #if found then replace that word with 'bad'

        l.append(j)                     #append it into a list

    z.append(l)                         #append the inner list to main list inorder to create list of list
train['tweets']=z  #create a new column for the processed tweets
print(train.tweet[26])

print(train.tweets[26])

#we can see that the first term from tweet feature is replaced with bad in tweets feature
l=[]

for i in train['tweets']:

    l.append(" ".join(i))

train['tweet_r']=l  #create new feature in which we stitch back the tokenized tweets
def process_tweets(train):

    train['mail_tweets']=train['tweet_r'].str.replace('https?://[A-Za-z0-9./]+',' ')#replace all the url links

    train['clean_tweets']=train['tweet_r'].str.replace('[^a-zA-Z]',' ')#replacing special characters with space

process_tweets(train)
train.head() #our cleaned train data.
train.drop(columns=['tweet','tweets','id','tweet_r','mail_tweets'],inplace=True)

#we will drop columns except 'clean_tweets' and 'label'
#Visualizing all the positive tweets using wordcloud plot.

positive=' '.join([text for text in train['clean_tweets'][train['label']==0]])

wordcloud=WordCloud(width=800,height=400,random_state=21,max_font_size=110).generate(positive)

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.title('positive tweets')

plt.show()
#Visualizing all the negative tweets using wordcloud plot.

positive=' '.join([text for text in train['clean_tweets'][train['label']==1]])

wordcloud=WordCloud(width=800,height=400,random_state=21,max_font_size=110).generate(positive)

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.title('negative tweets')

plt.show()
train['label'].value_counts().plot(kind='barh')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2,max_features=1000, stop_words='english') 

bow = bow_vectorizer.fit_transform(train['clean_tweets']) 

bow.shape
tdf_vectorize=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000)

tfidf = tdf_vectorize.fit_transform(train['clean_tweets'])

tfidf.shape
from sklearn.linear_model import LogisticRegression #import LinerRegression from sklearn

from sklearn.model_selection import train_test_split 

from sklearn.metrics import f1_score
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(bow,train['label'],random_state=42,test_size=0.3)
xtrain_tdf, xvalid_tdf, ytrain, yvalid = train_test_split(tfidf, train['label'],random_state=42,test_size=0.3)
#using Bag of word features

lr=LogisticRegression()

lr.fit(xtrain_bow,ytrain)

lr.score(xvalid_bow,yvalid)
#using TF-IDF features

lr=LogisticRegression()

lr.fit(xtrain_tdf,ytrain)

lr.score(xvalid_tdf,yvalid)