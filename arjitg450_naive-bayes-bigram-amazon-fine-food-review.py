# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve,auc

from nltk.stem.porter import PorterStemmer

 
#using SQLite table to read data

con  =sqlite3.connect("/kaggle/input/amazon-fine-food-reviews/database.sqlite")
filtered_data = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3

""",con)









def partition(x):

    if x < 3:

        return "negative"

    return "positive"







actScore=filtered_data["Score"]



positiveNegative = filtered_data["Score"].map(partition)

filtered_data["Score"] = positiveNegative

#print(filtered_data["Score"])



filtered_data.shape



filtered_data.head(10)



display = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId = "AR5J8UI46CURR"

ORDER BY ProductID

""",con)

display



sorted_data = filtered_data.sort_values("ProductId",axis=0,ascending=True)



final= sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep="first",inplace=False)



final.shape



filtered_data.shape



final = final[final["HelpfulnessNumerator"]<=final["HelpfulnessDenominator"]]

final.shape



#final["Time"].value_counts()



display = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Time = "1350345600"

ORDER BY ProductID

""",con)

#display

#noissue in Time

final=final[0:50000]

final.shape

final.head(3)
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from bs4 import BeautifulSoup

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





#stop = set(stopwords.words("english"))









# Combining all the above stundents 

from tqdm import tqdm

preprocessed_reviews = []

# tqdm is for printing the status bar

for sentance in tqdm(final['Text'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split())

    preprocessed_reviews.append(sentance.strip())

    #print(preprocessed_reviews)

    for e in sentance.split():

        sentance=' '.join(e.lower())

       #     print(e)

        
from sklearn.model_selection import train_test_split

#print(preprocessed_reviews[40001])

print(len(preprocessed_reviews))



final["Text"]=preprocessed_reviews

final["Score"]=final["Score"].replace("positive",1)

final["Score"]=final["Score"].replace("negative",0)

final.head(20)

#print(final.iloc[ t : t+1 ,  6 : 7 ])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final['Text'], final["Score"], random_state=1)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,ngram_range=(2,2),min_df=10, max_features=5000)

X_train_cv = cv.fit_transform(X_train)

X_test_cv = cv.transform(X_test)

#y_train_cv = cv.fit_transform(y_train)

#y_test_cv = cv.transform(y_test)

#cont_vector=CountVectorizer(ngram_range=(2,2),min_df=10, max_features=5000)

 
from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

naive_bayes.fit(X_train_cv, y_train)

predictions = naive_bayes.predict(X_test_cv)
from sklearn.metrics import accuracy_score, precision_score, recall_score

print('Accuracy score: ', accuracy_score(y_test, predictions))

print('Precision score: ', precision_score(y_test, predictions))

print('Recall score: ', recall_score(y_test, predictions))
#fi_bigram = cont_vector.fit(preprocessed_reviews) #doing only fit
#print("some feature names ", fi_bigram.get_feature_names()[0:20])


#fi_bigram.get_shape()
#fi_bigram_transformed[1]
#tfidf

'''

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(preprocessed_reviews)

print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[100:110])

print('='*50)



final_tf_idf = tf_idf_vect.transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_tf_idf))

print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1]) '''
# Train your own Word2Vec model using your own text corpus

'''i=0

list_of_sentance=[]

for sentance in preprocessed_reviews:

    list_of_sentance.append(sentance.split()) '''
# min_count = 5 considers only words that occured atleast 5 times

'''w2v_model=Word2Vec(list_of_sentance ,min_count=5,size=50, workers=4)

print(w2v_model.wv.most_similar('great'))

print('='*50) 

print(w2v_model.wv.most_similar('worst')) '''
'''w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum 5 times ",len(w2v_words))

print("sample words ", w2v_words[0:50])'''
'''# average Word2Vec

# compute average word2vec for each review.

sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list

for sent in tqdm(list_of_sentance): # for each review/sentence

    sent_vec = np.zeros(50) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors.append(sent_vec)

print(len(sent_vectors))

print(len(sent_vectors[0])) '''
# S = ["abc def pqr", "def def def abc", "pqr pqr def"]

'''model = TfidfVectorizer()

model.fit(preprocessed_reviews)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_))) '''
#naive bayes