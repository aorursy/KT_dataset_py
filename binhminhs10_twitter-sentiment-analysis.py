# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/tweets"))



# Any results you write to the current directory are saved as output.
import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import string

import nltk
train  = pd.read_csv('../input/tweets/train_E6oV3lV.csv')

test = pd.read_csv('../input/tweets/test_tweets_anuFYb8.csv')
train.head()
combi = train.append(test, ignore_index=True)
def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt  
# remove twitter handles (@user)

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
combi.head()
# remove special characters, numbers, punctuations

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi.head(10)
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

tokenized_tweet.head()
from nltk.stem.porter import *

stemmer = PorterStemmer()



tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

tokenized_tweet.head()
for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])



combi['tidy_tweet'] = tokenized_tweet

combi.head()
all_words = ' '.join([text for text in combi['tidy_tweet']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

wordcloud = WordCloud(width=800, height=500,

random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# function to collect hashtags

def hashtag_extract(x):

    hashtags = []

    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags
# extracting hashtags from non racist/sexist tweets



HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])



# extracting hashtags from racist/sexist tweets

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])



# unnesting list

HT_regular = sum(HT_regular,[])

HT_negative = sum(HT_negative,[])

a = nltk.FreqDist(HT_regular)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
b = nltk.FreqDist(HT_negative)

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags

e = e.nlargest(columns="Count", n = 10)   

plt.figure(figsize=(16,5))

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# bag-of-words feature matrix

bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
import gensim

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing



model_w2v = gensim.models.Word2Vec(

            tokenized_tweet,

            size=200, # desired no. of features/independent variables 

            window=5, # context window size

            min_count=2,

            sg = 1, # 1 for skip-gram model

            hs = 0,

            negative = 10, # for negative sampling

            workers= 2, # no.of cores

            seed = 34)



model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)
def word_vector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0.

    for word in tokens:

        try:

            vec += model_w2v[word].reshape((1, size))

            count += 1.

        except KeyError: # handling the case where the token is not in vocabulary

                         

            continue

    if count != 0:

        vec /= count

    return vec
wordvec_arrays = np.zeros((len(tokenized_tweet), 200))



for i in range(len(tokenized_tweet)):

    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)

    

wordvec_df = pd.DataFrame(wordvec_arrays)

wordvec_df.shape



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



train_bow = bow[:31962,:]

test_bow = bow[31962:,:]



# splitting data into training and validation set

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)



lreg = LogisticRegression()

lreg.fit(xtrain_bow, ytrain) # training the model



prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set

prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

prediction_int = prediction_int.astype(np.int)



f1_score(yvalid, prediction_int) # calculating f1 score
train_tfidf = tfidf[:31962,:]

test_tfidf = tfidf[31962:,:]



xtrain_tfidf = train_tfidf[ytrain.index]

xvalid_tfidf = train_tfidf[yvalid.index]



lreg.fit(xtrain_tfidf, ytrain)



prediction = lreg.predict_proba(xvalid_tfidf)

prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)



f1_score(yvalid, prediction_int)
train_w2v = wordvec_df.iloc[:31962,:]

test_w2v = wordvec_df.iloc[31962:,:]



xtrain_w2v = train_w2v.iloc[ytrain.index,:]

xvalid_w2v = train_w2v.iloc[yvalid.index,:]
lreg.fit(xtrain_w2v, ytrain)



prediction = lreg.predict_proba(xvalid_w2v)

prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)