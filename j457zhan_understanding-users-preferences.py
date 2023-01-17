# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/warframe.csv")

df.head()
# Groups of playTime

#playTime = ["0-1","1-10","10-50","50-100","100-500","500-1000","1000-1500","1500-2000","2000-2500","2500-3000","3000+"]

playTime = ["0-1","1-10","10-50","50-100","100-500","500-1000","1000-3000","3000+"]

df['playTime'] = pd.cut(df['hours'],bins=[0,1,10,50,100,500,1000,3000,11000],labels=playTime)

df['playTime'] = df['playTime'].astype('str')
# Preprocessing

df_review = df[["date", "hours", "products","text", "recommended","playTime"]].copy()

df_review["recommended"] = df_review["recommended"].astype(dtype=np.int64)

df_review["text"] = df_review["text"].astype(str)

df_review.head()
printable = set(string.printable)

printable.remove("'")

numbers = []

for i in range(10):

    numbers.append(str(i))

for number in numbers:

    printable.remove(number)



def pre_process(text):

    text = text.lower()

    text = re.sub('&lt;/?.*?&gt;',' &lt;&gt; ',text)

    text=re.sub('(\\d|\\W)+',' ',text)

    

    return text

df_review['cleantext'] = df_review['text'].apply(lambda row: ''.join(filter(lambda x:x in printable,row)))

df_review['cleantext'] = df_review['cleantext'].apply(lambda x:pre_process(x))

df_review.head()
from nltk.corpus import stopwords

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS



english_stop_words = ENGLISH_STOP_WORDS

def remove_stop_words(corpus):

    removed_stop_words = []

    for review in corpus:

        removed_stop_words.append(

            ' '.join([word for word in review.split() 

                      if word not in english_stop_words])

        )

    return removed_stop_words



df_review['cleantext'] = remove_stop_words(df_review['cleantext'])

df_review.head()
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer



def get_stemmed_text(corpus):

    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

def get_lemmatized_text(corpus):

    lemmatizer = WordNetLemmatizer()

    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]



df_review['stemmedtext'] = get_stemmed_text(df_review['cleantext'])

df_review['lemmatext'] = get_lemmatized_text(df_review['stemmedtext'])

df_review.head()
# tf-idf weighting

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf_vectorizer.fit(df_review['lemmatext'])

X = tfidf_vectorizer.transform(df_review['lemmatext'])

y = df_review['recommended']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    

    lr = LogisticRegression(C=c)

    lr.fit(X_train, y_train)

    print('Accuracy for C=%s: %s' %(c, accuracy_score(y_test, lr.predict(X_test))))

final_tfidf = LogisticRegression(C=1)

final_tfidf.fit(X_train, y_train)

accuracy_score(y_test, final_tfidf.predict(X_test))
# Determine the output(Overall, Single-Word, Paired-Words)

def sum_ngram(ngram_dict):

    positive = []

    for best_positive in sorted(

        ngram_dict.items(),

        key=lambda x: x[1],

        reverse=True)[:20]:

        positive.append(best_positive)

    positive = pd.DataFrame(positive)

    positive.columns = ["poswords", "pos_importance"]

    

    negative = []

    for best_negative in sorted(

        ngram_dict.items(),

        key=lambda x: x[1])[:20]:

        negative.append(best_negative)

    negative = pd.DataFrame(negative)

    negative.columns = ["negwords", "neg_importance"]

    total = positive.join(negative)

    return total



# tfidf Model

def tfidf_game_model(hrs, ngram):

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    tfidf_vectorizer.fit(df_review.loc[df_review["playTime"]==hrs]['lemmatext'])

    X = tfidf_vectorizer.transform(df_review.loc[df_review["playTime"]==hrs]['lemmatext'])

    y = df_review.loc[df_review["playTime"]==hrs]['recommended']



    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    

    final_tfidf = LogisticRegression(C=1)

    final_tfidf.fit(X_train, y_train)

    

    tfidf_feature_to_coef = {

    word: coef for word, coef in zip(

     tfidf_vectorizer.get_feature_names(), final_tfidf.coef_[0])

}

    single_tfidf_feature_to_coef={}

    pair_tfidf_feature_to_coef={}

    for key, val in tfidf_feature_to_coef.items():

        if len(key.split()) == 1:

            single_tfidf_feature_to_coef[key] = val

        else:

            pair_tfidf_feature_to_coef[key] = val

            

    # return Overall/Single/Paired Analysis        

    if ngram == None:

        return(sum_ngram(tfidf_feature_to_coef))

    if ngram == 1:

        return(sum_ngram(single_tfidf_feature_to_coef))

    else:

        return(sum_ngram(pair_tfidf_feature_to_coef))
df_review.head()
tfidf_game_model("0-1",ngram=1)
tfidf_game_model("0-1",ngram=2)
tfidf_game_model("1-10",ngram=1)
tfidf_game_model("1-10",ngram=2)
tfidf_game_model("10-50",ngram=1)
tfidf_game_model("10-50",ngram=2)
tfidf_game_model("50-100",ngram=1)
tfidf_game_model("50-100",ngram=2)
tfidf_game_model("100-500",ngram=1)
tfidf_game_model("100-500",ngram=2)
tfidf_game_model("500-1000",ngram=1)
tfidf_game_model("500-1000",ngram=2)
tfidf_game_model("1000-3000",ngram=1)
tfidf_game_model("1000-3000",ngram=2)
tfidf_game_model("3000+",ngram=1)
tfidf_game_model("3000+",ngram=2)