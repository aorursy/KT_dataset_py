# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.0
from nltk import word_tokenize

import nltk



stop_words = nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')



train.head()
train.author.unique()
test.head()
sample.head()
train.text[:10]
print("Input data has {} rows and {} columns".format(len(train), 

                                                    len(train.columns)))
print("Out of {} rows, {} are EAP, {} are HPL, {} are MWS".format(len(train), 

                                                                 len(train[train["author"]== "EAP"]),

                                                                 len(train[train["author"]=="HPL"]), 

                                                                 len(train[train["author"]=="MWS"])))
train.isnull().sum()
import string

import re



def cleaned_text(text):

    text_nopunct = "".join([char.lower() for char in text if char not in string.punctuation])

    tokenized = re.split("\W+", text_nopunct)

    stem_text = [ps.stem(word) for word in tokenized if word not in stop_words]

    return stem_text 



train["text_cleaned"] = train["text"].apply(lambda x:cleaned_text(x))

test["text_cleaned"] = test["text"].apply(lambda x:cleaned_text(x))

train.head()
test.head()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(analyzer=cleaned_text)

x_tfidf = tfidf.fit(train["text_cleaned"])



tfidf_train = x_tfidf.transform(train["text_cleaned"])

tfidf_test = x_tfidf.transform(test["text_cleaned"]) 



print(tfidf_train.shape)

print(tfidf_test.shape)

#print(tfidf.get_feature_names())
df_tfidf_train = pd.DataFrame(tfidf_train.toarray())

df_tfidf_test = pd.DataFrame(tfidf_test.toarray())



df_tfidf_train.columns = tfidf.get_feature_names()

df_tfidf_test.columns = tfidf.get_feature_names()



df_tfidf_train.head()
def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round((count/(len(text)-text.count(" ")))*100, 3)



train["body_length"] = train["text"].apply(lambda x:len(x)-x.count(" "))

train["punctuation%"] = train["text"].apply(lambda x: count_punc(x))



test["body_length"] = test["text"].apply(lambda x:len(x)-x.count(" "))

test["punctuation%"] = test["text"].apply(lambda x: count_punc(x))



train.head()
test.head()
import matplotlib.pyplot as plt



bins = np.linspace(0,200, 40)

plt.hist(train[train["author"]=="EAP"]["body_length"], bins, alpha=0.5, normed=True, label="EAP")

plt.hist(train[train["author"]=='HPL']["body_length"], bins, alpha=0.5, normed=True, label='HPL')

plt.hist(train[train["author"]=='MWS']["body_length"], bins, alpha=0.5, normed=True, label='MWS')

plt.legend(loc="best")

plt.show()
bins = np.linspace(0,200, 40)

plt.hist(train[train["author"]=="EAP"]["punctuation%"], bins, alpha=0.5, normed=True, label="EAP")

plt.hist(train[train["author"]=='HPL']["punctuation%"], bins, alpha=0.5, normed=True, label='HPL')

plt.hist(train[train["author"]=='MWS']["punctuation%"], bins, alpha=0.5, normed=True, label='MWS')

plt.legend(loc="best")

plt.show()
df_train = pd.concat([train[["body_length", "punctuation%"]].reset_index(drop=True), df_tfidf_train], axis=1)

df_test = pd.concat([test[["body_length", "punctuation%"]].reset_index(drop=True), df_tfidf_test], axis=1)

df_train.head()
df_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(df_train, train["author"], test_size=0.2)



def train_rf(n_est, depth):

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)

    rf_model = rf.fit(x_train, y_train)

    y_pred = rf_model.predict(x_test)

    score = accuracy_score(y_test, y_pred)

    print("n_estimator: {} / depth: {} / accuracy score: {}".format(n_est, depth, score))

    

for i in [10,50,100,150]:

    for k in [10,20,30,40,None]:

        train_rf(i, k)
rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)

rf_model = rf.fit(df_train, train["author"])

y_pred = rf_model.predict(df_test)
df_pro = rf.predict_proba(df_test)

df_prob = pd.DataFrame(df_pro)

df_prob.columns = ["EAP","HPL", "MWS"]

df_prob.head()



df = pd.concat([test["id"], df_prob], axis=1)

df.head(10)