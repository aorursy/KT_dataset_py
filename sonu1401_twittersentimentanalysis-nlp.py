import re # for regular expressions

import pandas as pd 

pd.set_option("display.max_colwidth", 200)

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import string

import nltk # for text manipulation

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)



%matplotlib inline
df = pd.read_csv("../input/TwitterSentimentAnalysis.csv")
df.shape
df.head()
df['label'].value_counts()
df.iloc[0:2,2]
df.rename(columns={'tweet_text':'tweet'},inplace=True)
##  Basic feature extraction using text data 

# Number of words

# Number of characters

# Average word length

# Number of stopwords

# Number of special characters

# Number of numerics

# Number of uppercase words
df.info()
# Number of words

df['Word_Count']=df['tweet'].apply(lambda x:len(str(x).split(" ")))
df.head()     
# Number of characters (means calculating the length of the tweet.)

df['char_count']=df.tweet.str.len()
df.head()  
# Average word length

def avg_word(sentence):

    words=sentence.split()

    return (sum(len(word) for word in words)/len(words))
df['Avg_word_length']=df.tweet.apply(lambda x:avg_word(x))
df.head()
# Number of stopwords

# Stopwords are individual words such as and, is , am ,are etc.., it is available under nltk.corpus package

from nltk.corpus import stopwords

stop=stopwords.words("english")

df['stop_word_count']=df['tweet'].apply(lambda x:len([x for x in x.split() if x in stop]))
df.head(3)
# Number of special characters (say number of HashTags)

# Number of special characters

# One more interesting feature which we can extract from a tweet is calculating the number of hashtags or mentions present in it. This also helps in extracting extra information from our text data.

# Here, we make use of the ‘starts with’ function because hashtags (or mentions) always appear at the beginning of a word.
df['HashTags']=df.tweet.apply(lambda x:len([x for x in x.split() if x.startswith('#')]))
df.head(3)
#Number of numerics

df['numerics']=df.tweet.apply(lambda x:len([x for x in x.split() if x.isdigit()]))
df.head(2)
# Number of Uppercase words

df['upperword']=df.tweet.apply(lambda x:len([x for x in x.split() if x.isupper()]))
df.head()
# Basic Text Pre-processing of text data 

# Lower casing

# Punctuation removal

# Stopwords removal

# Frequent words removal

# Rare words removal

# Spelling correction

# Tokenization

# Stemming

# Lemmatization
# So far, we have learned how to extract basic features from text data.

# Before diving into text and feature extraction, our first step should be cleaning the data in order to obtain better features.

# We will achieve this by doing some of the basic pre-processing steps on our training data.
df['tweet']=df.tweet.apply(lambda x:" ".join(x.lower() for x in x.split()))
df['tweet'].head()
df['tweet']=df.tweet.str.replace('[^\w\s]','')
from nltk.corpus import stopwords

stop=stopwords.words("english")
df['tweet']=df.tweet.apply(lambda x:" ".join(x for x in x.split() if x not in stop))
df.head()
freq = pd.Series(' '.join(df['tweet']).split()).value_counts()[:10]

freq
freq = list(freq.index)

freq
df.head(3)
df['tweet']=df.tweet.apply(lambda x:" ".join(x for x in x.split() if x not in freq))
df.head(3)
freq = pd.Series(" ".join(df['tweet']).split()).value_counts()[-10:]
freq
freq = list(freq.index)

df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

df['tweet'].head()
# In that regard, spelling correction is a useful pre-processing step because this also will help us in reducing multiple

# copies of words. For example, “Analytics” and “analytcs” will be treated as different words even if they are used in the

# same sense.

# To achieve this we will use the textblob library
from textblob import TextBlob
from textblob import TextBlob

df['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))
df['tokenized_tweet'] = df['tweet'].apply(lambda x: x.split()) 
df.head()
from nltk.stem.porter import *

stemmer = PorterStemmer()

df['stemmed_text'] = df['tweet'].apply(lambda x: [stemmer.stem(i) for i in x]) 
df.iloc[:,5:15].head()
from nltk.stem.wordnet import WordNetLemmatizer

lem=WordNetLemmatizer()



df['Lemmatized_word'] = df.tokenized_tweet.apply(lambda x: [lem.lemmatize(i) for i in x]) 
df.iloc[:,10:16].head()
# N-grams

# N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams.

# Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used. Unigrams do not usually contain

# as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that 

# they capture the language structure, like what letter or word is likely to follow the given one. The 

# longer the n-gram (the higher the n), the more context you have to work with. Optimum length really 

# depends on the application – if your n-grams are too short, you may fail to capture important differences. 

# On the other hand, if they are too long, you may fail to capture the “general knowledge” and only stick to

# particular cases.
def generate_ngrams(text, n):

    words = text.split()

    output = []  

    for i in range(len(words)-n+1):

        output.append(words[i:i+n])

    return output
generate_ngrams('this is a sample text', 2)
# This is in case of our dataframe

TextBlob(df['tweet'][0]).ngrams(2)
df.info()
import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt
all_words = ' '.join(text for text in df['tweet'].astype(str))

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=2500,

                      height=2000

                     ).generate(all_words)

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
all_words = ' '.join(text for text in df['tweet'].astype(str))

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=2500,

                      height=2000

                     ).generate(all_words)

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
all_words = ' '.join(text for text in df['tokenized_tweet'].astype(str))

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=2500,

                      height=2000

                     ).generate(all_words)

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
all_words = ' '.join(text for text in df['stemmed_text'].astype(str))

from wordcloud import WordCloud

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=2500,

                      height=2000

                     ).generate(all_words)

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
all_words = ' '.join(text for text in df['Lemmatized_word'].astype(str))

from wordcloud import WordCloud

wordcloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
df.shape
df.head(1)
len(df['Lemmatized_word'])
df1=df[['id','label','tweet','Lemmatized_word']]
df1['label'].value_counts()
df1.head()
df1.values
words_lem=[]

for i in df['Lemmatized_word'].astype(str):

    WNlemma = nltk.WordNetLemmatizer()

    words_lem.append(WNlemma.lemmatize(i))
x=[]

for j in words_lem:

    for i in df1.tweet:

        if j in i:

            x.append(1)

        else:

            x.append(0)
import numpy as np

x=np.array(x)
len(x)
y=df['label']

x=df['tweet']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=101)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.head(2)
from sklearn.feature_extraction.text import CountVectorizer



# Fit the CountVectorizer to the training data

vect = CountVectorizer().fit(X_train)

# transform the documents in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)
X_train_vectorized
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
# kNN Classification

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=2)

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
# Naive Bayes

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
# Random Forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
# Bagging Classifier

from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))