import numpy as np

import pandas as pd 

import os

import gensim

print(os.listdir("../input"))

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

## Collection of all these word vectorings is embeddings
len(embeddings['modi'])
embeddings.most_similar('rahul', topn=10)
embeddings.most_similar('hyundai', topn=10)
embeddings.doesnt_match(['football','basketball','cricket','apple'])
url = 'https://bit.ly/2S2yXEd'

imdb = pd.read_csv(url)

imdb['review'].head()
import nltk

doc = imdb.loc[0, 'review']

words = nltk.word_tokenize(doc.lower())

temp =pd.DataFrame()

for word in words:

    try:

        print(word, embeddings[word][:5])

        temp = temp.append(pd.Series(embeddings[word]), ignore_index = True)

    except:

        print(word, 'is not there')

temp
docs = imdb['review'].str.replace('-',' ').str.lower().str.replace('[^a-z ]', '')

stopwords = nltk.corpus.stopwords.words('english')

clean_sentence = lambda doc: ' '.join([word for word in nltk.word_tokenize(doc) if word not in stopwords])

docs_clean = docs.apply(clean_sentence)

docs_clean.head()
docs_vectors = pd.DataFrame()

for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec), ignore_index=True)

        except:

            pass

    docs_vectors = docs_vectors.append(temp.mean(), ignore_index = True)

docs_vectors.shape
imdb.isnull().sum()
imdb.loc[64, 'review']
imdb.loc[590, 'review']
X = docs_vectors.drop([64, 590])

y = imdb['sentiment'].drop([64, 590])

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,y, test_size = 0.2, random_state = 100)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=800).fit(train_x, train_y)

test_pred = model.predict(test_x)

accuracy_score(test_y, test_pred)
model = AdaBoostClassifier(n_estimators=800).fit(train_x, train_y)

test_pred = model.predict(test_x)

accuracy_score(test_y, test_pred)