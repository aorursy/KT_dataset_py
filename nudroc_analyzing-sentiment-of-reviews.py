from __future__ import division

from __future__ import print_function

import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab

from bs4 import BeautifulSoup

import textblob

from textblob import TextBlob

import datetime

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import time

from IPython.display import display
#read in Amazon Fine Food Reviews from Kaggle/SNAP

data = pd.read_csv("../input/Reviews.csv")

print (data.head())

print(data.dtypes)

%%time

data['text_cln']= data['Text'].map(lambda x: BeautifulSoup(x, "lxml").get_text())
%%time

data['tb_polarity']= data['text_cln'].map(lambda x: TextBlob(x).sentiment.polarity)
#normalize date time

data2 = data.copy()

data2['datetime'] = data2['Time'].map(lambda x: (datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')))

data2['datetime'] = pd.to_datetime(data2['datetime'])
display(data2.head())
print(data2.dtypes)

display(data2.describe())

#distribution of tb_polarity, mean = 0.241566

data2['tb_polarity'].hist(bins=25)
timecopy = data2.copy()

timecopy.set_index('datetime', inplace=True)



time2012 = timecopy.loc['2012-1-01':'2012-12-31']

s1 = [time2012['tb_polarity'].mean(), time2012['Score'].mean(), time2012['HelpfulnessNumerator'].mean(),

     time2012['HelpfulnessDenominator'].mean(), len(time2012),time2012['ProductId'].nunique()]



time2011 = timecopy.loc['2011-1-01':'2011-12-31']

s2 = [time2011['tb_polarity'].mean(), time2011['Score'].mean(), time2011['HelpfulnessNumerator'].mean(),

     time2011['HelpfulnessDenominator'].mean(), len(time2011), time2011['ProductId'].nunique()]



time2010 = timecopy.loc['2010-1-01':'2010-12-31']

s3 = [time2010['tb_polarity'].mean(), time2010['Score'].mean(), time2010['HelpfulnessNumerator'].mean(),

     time2010['HelpfulnessDenominator'].mean(), len(time2010), time2010['ProductId'].nunique()]



time2009 = timecopy.loc['2009-1-01':'2009-12-31']

s4 = [time2009['tb_polarity'].mean(), time2009['Score'].mean(), time2009['HelpfulnessNumerator'].mean(),

     time2009['HelpfulnessDenominator'].mean(), len(time2009), time2009['ProductId'].nunique()]



time2008 = timecopy.loc['2008-1-01':'2008-12-31']

s5 = [time2008['tb_polarity'].mean(), time2008['Score'].mean(), time2008['HelpfulnessNumerator'].mean(),

     time2008['HelpfulnessDenominator'].mean(), len(time2008), time2008['ProductId'].nunique()]

df = pd.DataFrame([s5,s4, s3, s2, s1], index=['2008', '2009', '2010', '2011', '2012'], columns =['mean polarity', 

                  'mean Score', 'mean HNum', 'mean HDen', 'count', 'num_products'])

df.plot(subplots=True, marker='*', figsize=(12,8))

std = []

def stddevframe(df):

    a = [df['tb_polarity'].std(), df['Score'].std(), df['HelpfulnessNumerator'].std(),

     df['HelpfulnessDenominator'].std(), len(df), df['ProductId'].nunique()]

    std.append(a)

    

stddevframe(time2008)

stddevframe(time2009)

stddevframe(time2010)

stddevframe(time2011)

stddevframe(time2012)
df_std = pd.DataFrame(std, index=['2008', '2009', '2010', '2011', '2012'], columns =['std polarity', 

                  'std Score', 'std HNum', 'std HDen', 'count', 'num_products'])

df_std.plot(subplots=True, marker='*', figsize=(12,8))
#polarity by ProductID

data_pid = data2.groupby(['ProductId'])['tb_polarity', 'Score'].mean().reset_index()



#polarity by Profile Name

data_pn = data2.groupby(['ProfileName'])['tb_polarity', 'Score'].mean().reset_index()



#Perfect Review Sets

pf_posr = data2.loc[(data2['tb_polarity'] == 1.0)]

pf_negr = data2.loc[(data2['tb_polarity'] == -1.0)]



#Perfect Review by ProductID

pf_pos_pid = data_pid.loc[(data_pid['tb_polarity'] == 1.0)]

pf_neg_pid = data_pid.loc[(data_pid['tb_polarity'] == -1.0)]

print('The number of perfectly positive sentiment reviews are',len(pf_posr))

print("%.4f"%(len(pf_posr)/len(data2)), 'of all reviews have 1.0 positive sentiment')
print('The number of perfectly negative sentiment reviews are', len(pf_negr))

print("%.4f"%(len(pf_negr)/len(data2)), 'of all reviews have -1.0 negative sentiment')
print('Products with perfectly positive average reviews are', len(pf_pos_pid))

print("%.4f"%(len(pf_pos_pid)/data2['ProductId'].nunique()), 'of all products have a mean 1.0 positive sentiment')
print('Products with perfectly negative average reviews are', len(pf_neg_pid))

print("%.4f"%(len(pf_neg_pid)/data2['ProductId'].nunique()), 'of all products have a mean -1.0 negative sentiment')
#Generating feature set: 

count_vect = CountVectorizer(decode_error='ignore', strip_accents='unicode', stop_words='english')

X_train_counts = count_vect.fit_transform(pf_posr['text_cln'])

feature_names = count_vect.get_feature_names()



#TF-IDF

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#Training model using NBM

clf_1 = MultinomialNB().fit(X_train_tfidf, pf_posr['Score'])



#Sort the coef_ as per feature weights and select largest 25 of them

def topfeatures_positive(score):

    inds = np.argsort(clf_1.coef_[score, :])[-25:]

    print("The top 20 most informative words for category:", score, "which is a", score+1, "star rating:")

    for i in inds: 

        f = feature_names[i]

        c = clf_1.coef_[score,[i]]

        print(f,c)
print(topfeatures_positive(0)) #1-star rating
print(topfeatures_positive(1)) #2-star rating

print(topfeatures_positive(2)) #3-star rating

print(topfeatures_positive(3)) #4-star rating

print(topfeatures_positive(4)) #5-star rating
count_vect = CountVectorizer(decode_error='ignore', strip_accents='unicode', stop_words='english')

X_train_counts = count_vect.fit_transform(pf_negr['text_cln'])

feature_names = count_vect.get_feature_names()



#TF-IDF

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#Training model using NBM

clf_2 = MultinomialNB().fit(X_train_tfidf, pf_negr['Score'])



#Sort the coef_ as per feature weights and select largest 25 of them

def topfeatures_negative(score):

    inds = np.argsort(clf_2.coef_[score, :])[-25:]

    print("The top 20 most informative words for category:", score, "which is a", score+1, "star rating:")

    for i in inds: 

        f = feature_names[i]

        c = clf_2.coef_[score,[i]]

        print(f,c)
print(topfeatures_negative(0)) #1-star rating

print(topfeatures_negative(1)) #2-star rating

print(topfeatures_negative(2)) #3-star rating

print(topfeatures_negative(3)) #4-star rating

print(topfeatures_negative(4)) #5-star rating
#For all comments with perfectly 1.0 positive sentiment

#tokenize and remove punctuation

copy1 = pf_posr[['ProductId', 'text_cln', 'Score']].copy()



#tokenize

tokenizer = RegexpTokenizer(r'\w+')

copy1['text_cln'] = copy1['text_cln'].map(lambda x: tokenizer.tokenize(x))

cachedstopwords = stopwords.words('english')

stw_set = set(cachedstopwords)

copy1['_cln'] = copy1.apply(lambda row: [item for item in row['text_cln'] if item not in stw_set], axis=1)



#sum up all comments where the Score is 5 Stars

copy1sum = ((copy1.loc[(copy1['Score'] == 5)]).reset_index())._cln.sum()
print(copy1sum[:25])
#For all comments with perfectly -1.0 negative sentiment

copy2 = pf_negr[['ProductId', 'text_cln', 'Score']].copy()

copy2['text_cln'] = copy2['text_cln'].map(lambda x: tokenizer.tokenize(x))

copy2['_cln'] = copy2.apply(lambda row: [item for item in row['text_cln'] if item not in stw_set], axis=1)

#sum up all comments where the Score is 1 Star

copy2sum = ((copy2.loc[(copy2['Score'] == 1)]).reset_index())._cln.sum()

print(copy2sum[:25])
def trigram(sum_text):

    tgm = nltk.collocations.TrigramAssocMeasures()

    finder = nltk.collocations.TrigramCollocationFinder.from_words(sum_text)

    scored = finder.score_ngrams(tgm.pmi)

    phrases = pd.DataFrame(scored)

    print(phrases)

    phrases.hist(bins=25)

    plt.show()



print(trigram(copy1sum))
print(trigram(copy2sum))