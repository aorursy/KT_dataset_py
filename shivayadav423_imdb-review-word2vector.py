



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings= gensim.models.KeyedVectors.load_word2vec_format(path, binary =True)
url = "https://raw.githubusercontent.com/skathirmani/datasets/master/imdb_sentiment.csv"

imdb= pd.read_csv(url)
imdb.head()
import nltk

doc = imdb.loc[0,"review"]

words = nltk.word_tokenize(doc.lower())

temp= pd.DataFrame()

for word in words:

    try:

        print(word,embeddings[word][:5])

        temp=temp.append(pd.Series(embeddings[word]),ignore_index=True)

    except:

        print(word,"is not there")
temp
docs = imdb["review"].str.replace("-"," ").str.lower().str.replace("[^a-z ]","")

docs.head()
stopwords=nltk.corpus.stopwords.words("english")

def clean_sentance(doc):

    words = doc.split(" ")

    words_clean = [word for word in nltk.word_tokenize(doc) if word  not in stopwords]

    docs_clean= " ".join(words_clean)

    return docs_clean

docs_clean=docs.apply(clean_sentance)

docs_clean.head()
docs_vectors = pd.DataFrame()

for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec),ignore_index=True)

        except:

                pass

    docs_vectors= docs_vectors.append(temp.mean(),ignore_index=True)

docs_vectors.shape
docs_vectors
pd.isnull(docs_vectors).sum(axis=1).sort_values(ascending=False)
x = docs_vectors.drop([64,590])

y = imdb["sentiment"].drop([64,590])

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.metrics import accuracy_score

model_rf = RandomForestClassifier(n_estimators=300).fit(train_x,train_y)

test_pred = model_rf.predict(test_x)

print(accuracy_score(test_y,test_pred))
model_Ad = AdaBoostClassifier(n_estimators=300).fit(train_x,train_y)

test_pred = model_Ad.predict(test_x)

print(accuracy_score(test_y,test_pred))