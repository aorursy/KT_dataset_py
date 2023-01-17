import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gensim

import nltk

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB , BernoulliNB

from sklearn.metrics import accuracy_score

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))
path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path , binary = True)

# Collection of words are listed in embeddings

# Default is 8 GB, shorter version will be provided

# Black box model

# For every word there are 300 lists
embeddings['amazon']

len(embeddings['amazon'])
embeddings.most_similar('rahul' , topn = 10)
embeddings.most_similar('hyundai' , topn = 10)
embeddings.doesnt_match(['football' , 'basketball' , 'cricket' , 'apple'])

# Cosine similarity among football , basketball , cricket are very high hence apple is given as odd man out
url = 'https://raw.githubusercontent.com/skathirmani/datasets/master/imdb_sentiment.csv'

df_imdb = pd.read_csv(url)

df_imdb['review'].head()
# one option is to add vectors with their elements

# Other option is to take average

# weights are the embeddings and are calculated in deep learning

#for the the 1st document a temporary df is created: first convert the document to a word-weight matrix and then compute column average

# the temporary df is created for all the documents in the corpus

# the final df contains the column weights for all the documents

# the # of columns are always 300 in the final df

# the #of rows depends on the input
df_imdb.loc[0]
doc = df_imdb.loc[0 , 'review']

words = nltk.word_tokenize(doc.lower())

temp = pd.DataFrame()

for word in words:

    try:

        print(embeddings[word][:5])

        temp = temp.append(pd.Series(embeddings[word]) , ignore_index = True)

    except:

        print(word, 'is not there')

temp

    
temp.mean()
docs = df_imdb['review'].str.replace('-' , ' ').str.lower().str.replace('[^a-z ]' , '')

StopWords = nltk.corpus.stopwords.words('english')

clean_sentence = lambda doc: ' '.join([word for word in nltk.word_tokenize(doc) if word not in StopWords])

docs_clean = docs.apply(clean_sentence)

docs_clean.head()
# Final DF

docs_vectors = pd.DataFrame() # document-Term Matrix

for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec) , ignore_index = True)

        except:

            pass

    docs_vectors = docs_vectors.append(temp.mean() , ignore_index = True)

docs_vectors.shape
pd.isnull(docs_vectors).sum(axis = 1).sort_values(ascending = False)

# In 64 and 590th row, there are missing values 
df_imdb.loc[64 , 'review']
df_imdb.loc[590 , 'review']
# since 64th row and 590th row are numbers we are dropping those rows

x = docs_vectors.drop([64 , 590])

y = df_imdb['sentiment'].drop([64 , 590])
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 100)
model_rf = RandomForestClassifier(n_estimators = 800).fit(x_train , y_train)

test_pred_rf = model_rf.predict(x_test)

print(accuracy_score(y_test , test_pred_rf))
model_ab =AdaBoostClassifier(n_estimators = 800).fit(x_train , y_train)

test_pred_ab = model_ab.predict(x_test)

print(accuracy_score(y_test , test_pred_ab))
model_gnb = GaussianNB().fit(x_train , y_train)

test_pred_gnb = model_gnb.predict(x_test)

print(accuracy_score(y_test , test_pred_gnb))
model_bnb = BernoulliNB().fit(x_train , y_train)

test_pred_bnb = model_bnb.predict(x_test)

print(accuracy_score(y_test , test_pred_bnb))