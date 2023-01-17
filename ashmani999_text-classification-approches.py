# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

from nltk.corpus import stopwords

from collections import Counter

import re

from  nltk.stem import WordNetLemmatizer

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

PUNCT_TO_REMOVE = string.punctuation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
reviews=pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')

print(reviews.head())
reviews.info()
reviews.shape
print(len(set(reviews.Text)))
reviews=reviews.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

reviews=reviews.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
print(reviews.shape)

print(reviews['Score'].value_counts())
def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



  

  

  

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in stopwords.words('english')])



  

def remove_words(text):

    return " ".join([word for word in str(text).split() if word not in words])

  

def remove_common_rare_words(features):

    cnt=Counter()

    for text in features:

        for word in text.split():

            cnt[word] += 1

    counter=cnt.most_common()

    most_common=counter[:10]

    most_rare=counter[-10:]

    FREQWORDS = set([w for (w, wc) in most_common])

    RAREWORDS= set([w for (w, wc) in most_rare])

    global words

    words=FREQWORDS.union(RAREWORDS)

    features=features.apply(lambda text: remove_words(text))

    return features





def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



def stem_word(text):

    stemmer=WordNetLemmatizer()

    return " ".join([stemmer.lemmatize(word) for word in str(text).split()])



def remove_unwanted_tokens(text):

    pattern=re.compile(r'[^a-z]\S')

    return " ".join([ pattern.sub(r'',word) for word in str(text).split()])

  

def preprocess_text(features):

  #Lowar Casting

    features = features.str.lower()

    features=features.apply(lambda text: decontracted(text)) 

    features =features.apply(lambda text: remove_punctuation(text)) 

    features =features.apply(lambda text: remove_punctuation(text))

    features=features.apply(lambda text: remove_stopwords(text))

    features=remove_common_rare_words(features)

    features=features.apply(lambda text:remove_urls(text))

    features=features.apply(lambda text:remove_unwanted_tokens(text))

    features=features.apply(lambda text:stem_word(text))

  

    return features
features=preprocess_text(reviews.Text)

def vectorization(train,test):

    vectorizer = TfidfVectorizer()

    vectorizerFit = vectorizer.fit(train)

    x_train = vectorizer.transform(train).toarray()

    x_tets=vectorizer.transform(test).toarray()

    filename = 'wordVectorizer.sav'

    pickle.dump(vectorizerFit, open(filename, 'wb'))

    return x

def xy(score):

    if score <3:

        return 'Negative'

    elif score==3:

        return 'Neutral'

    return 'Positive'

label=reviews.Score.apply(lambda Score: xy(Score))
set(label)
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

labels=le.fit_transform(label)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)

X_train, X_test=vectorization(X_train, X_test)