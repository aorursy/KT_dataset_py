# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Game id in Steam

# AC 15100, AC2 33230, Brotherhood 48190, Revelations 201870

# AC3 208480, BlackFlag 242050, Rogue 311560, Unity 289650

# Syndicate 368500, Origins 582160, Odyssey 812140

df1 = pd.read_csv("../input/01_ac1.csv")

df2 = pd.read_csv("../input/02_ac2.csv")

df3 = pd.read_csv("../input/03_brotherhood.csv")

df4 = pd.read_csv("../input/04_revelations.csv")

df5 = pd.read_csv("../input/05_ac3.csv")

df6 = pd.read_csv("../input/06_blackflag.csv")

df7 = pd.read_csv("../input/07_rogue.csv")

df8 = pd.read_csv("../input/08_unity.csv")

df9 = pd.read_csv("../input/09_syndicate.csv")

df10 = pd.read_csv("../input/10_origins.csv")

df11 = pd.read_csv("../input/11_odyssey.csv")

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11], ignore_index=True)

df.replace({15100: "AC1", 33230: "AC2", 48190:"Brotherhood",

           201870: "Revelations", 208480: "AC3", 242050: "BlackFlag",

           311560: "Rogue", 289650: "Unity", 368500: "Syndicate",

           582160: "Origins", 812140: "Odyssey"}, inplace=True)

df.head()
df["product_id"].value_counts()
sns.set()

sns.catplot(x="product_id", y="hours", hue='recommended', kind="bar", data=df,aspect=3)
sns.set()

sns.catplot(x="product_id", y="products", hue='recommended', kind="bar", data=df,aspect=3)
df_review = df[["product_id", "text", "recommended"]].copy()

df_review["recommended"] = df_review["recommended"].astype(dtype=np.int64)

df_review["text"] = df_review["text"].astype(str)

df_review.head()
printable = set(string.printable)

# found a lot of words start with "_", so removed it

printable.remove("_")

printable.remove("'")



def pre_process(text):

    text = text.lower()

    text = re.sub('&lt;/?.*?&gt;',' &lt;&gt; ',text)

    text=re.sub('(\\d|\\W)+',' ',text)

    

    return text

df_review['cleantext'] = df_review['text'].apply(lambda row: ''.join(filter(lambda x:x in printable,row)))

df_review['cleantext'] = df_review['cleantext'].apply(lambda x:pre_process(x))
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
import random

random.seed(0)
# Single Word



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



gram_vectorizer = CountVectorizer(binary=True)

gram_vectorizer.fit(df_review['lemmatext'])

X = gram_vectorizer.transform(df_review['lemmatext'])

y = df_review['recommended']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    

    lr = LogisticRegression(C=c)

    lr.fit(X_train, y_train)

    print('Accuracy for C=%s: %s' % (c, accuracy_score(y_test, lr.predict(X_test))))

    
# Include paired words

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))

ngram_vectorizer.fit(df_review['lemmatext'])

X = ngram_vectorizer.transform(df_review['lemmatext'])

y = df_review['recommended']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)



for c in [0.01, 0.05, 0.25, 0.5, 1]:

    

    lr = LogisticRegression(C=c)

    lr.fit(X_train, y_train)

    print('Accuracy for C=%s: %s' % (c, accuracy_score(y_test, lr.predict(X_test))))
final_ngram = LogisticRegression(C=1)

final_ngram.fit(X_train, y_train)

print('Final Accuracy: %s' % accuracy_score(y_test, final_ngram.predict(X_test)))
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
cv_feature_to_coef = {

    word: coef for word, coef in zip(

     ngram_vectorizer.get_feature_names(), final_ngram.coef_[0])

}

tfidf_feature_to_coef = {

    word: coef for word, coef in zip(

     tfidf_vectorizer.get_feature_names(), final_tfidf.coef_[0])

}



print('Positive Words for CountVectorizer')

for best_positive in sorted(

    cv_feature_to_coef.items(),

    key=lambda x: x[1],

    reverse=True)[:10]:

    print(best_positive)



print('Positive Words for TfidfVectorizer')

for best_positive in sorted(

    tfidf_feature_to_coef.items(),

    key=lambda x: x[1],

    reverse=True)[:10]:

    print(best_positive)

    

print('Negative Words for CountVectorizer')

for best_negative in sorted(

    cv_feature_to_coef.items(),

    key=lambda x: x[1])[:10]:

    print(best_negative)

print('Negative Words for TfidfVectorizer')

for best_negative in sorted(

    tfidf_feature_to_coef.items(),

    key=lambda x: x[1])[:10]:

    print(best_negative)
games = np.unique(df_review["product_id"].values).tolist()



# CV Model

def cv_game_model(game):

    count_vectorizer = CountVectorizer(ngram_range=(1,2))

    count_vectorizer.fit(df_review.loc[df_review["product_id"]==game]['lemmatext'])

    X = count_vectorizer.transform(df_review.loc[df_review["product_id"]==game]['lemmatext'])

    y = df_review.loc[df_review["product_id"]==game]['recommended']



    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    

    final_cv = LogisticRegression(C=1)

    final_cv.fit(X_train, y_train)

    

    cv_feature_to_coef = {

    word: coef for word, coef in zip(

     count_vectorizer.get_feature_names(), final_cv.coef_[0])

}  

    positive = []

    for best_positive in sorted(

        cv_feature_to_coef.items(),

        key=lambda x: x[1],

        reverse=True)[:10]:

        positive.append(best_positive)

    positive = pd.DataFrame(positive)

    positive.columns = ["poswords", "pos_importance"]

    

    negative = []

    for best_negative in sorted(

        cv_feature_to_coef.items(),

        key=lambda x: x[1])[:10]:

        negative.append(best_negative)

    negative = pd.DataFrame(negative)

    negative.columns = ["negwords", "neg_importance"]

    total = positive.join(negative)

    return total

# Determine the output(Overall, Single-Word, Paired-Words)

def sum_ngram(ngram_dict):

    positive = []

    for best_positive in sorted(

        ngram_dict.items(),

        key=lambda x: x[1],

        reverse=True)[:10]:

        positive.append(best_positive)

    positive = pd.DataFrame(positive)

    positive.columns = ["poswords", "pos_importance"]

    

    negative = []

    for best_negative in sorted(

        ngram_dict.items(),

        key=lambda x: x[1])[:10]:

        negative.append(best_negative)

    negative = pd.DataFrame(negative)

    negative.columns = ["negwords", "neg_importance"]

    total = positive.join(negative)

    return total



# tfidf Model

def tfidf_game_model(game, ngram):

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    tfidf_vectorizer.fit(df_review.loc[df_review["product_id"]==game]['lemmatext'])

    X = tfidf_vectorizer.transform(df_review.loc[df_review["product_id"]==game]['lemmatext'])

    y = df_review.loc[df_review["product_id"]==game]['recommended']



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
tfidf_game_model("AC1",ngram=None)
tfidf_game_model("AC1",ngram=1)
tfidf_game_model("AC1",ngram=2)
tfidf_game_model("Origins", ngram=2)
tfidf_game_model("Odyssey",ngram=2)