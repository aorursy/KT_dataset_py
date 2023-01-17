# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gensim

#print(os.listdir("../input"))

print(os.listdir('../input/embeddings/GoogleNews-vectors-negative300/'))

# Any results you write to the current directory are saved as output.
path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path,binary = True)
len(embeddings['rahul'])
embeddings.most_similar('rahul',topn = 10) #top 10 words similar to rahul
embeddings.doesnt_match(['football','basketball','cricket','apple'])

#Cosine similarity is checked. Apple has the least wtr to other
url ='https://bit.ly/2S2yXEd'

imdb = pd.read_csv(url)
imdb['review'].head()
import nltk

doc = imdb.loc[0,'review']

words = nltk.word_tokenize(doc.lower())



temp = pd.DataFrame()

for word in words:

    try:

        print(embeddings[word][:5])

        temp = temp.append(pd.Series(embeddings[word]),ignore_index= True)

        #temp

    except:

        print(word,'does not have a vector representation')
docs = imdb['review'].str.replace('-',' ').str.lower().str.replace('[^a-z ]','') 

stopwords = nltk.corpus.stopwords.words('english')

clean_sentence = lambda doc: ' '.join([word for word in nltk.word_tokenize(doc) if word not in stopwords])

docs_clean = docs.apply(clean_sentence)

docs_vectors = pd.DataFrame()

for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp = pd.DataFrame()

    for word in words : 

        try: 

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec), ignore_index= True)

        except:

            pass #goes to the next word 

    docs_vectors = docs_vectors.append(temp.mean(), ignore_index = True)

docs_vectors.shape

    
docs_vectors.head()
X = docs_vectors.drop([64,590])

y = imdb['sentiment'].drop([64,590])



from sklearn.model_selection import train_test_split

train_x , test_x , train_y, test_y = train_test_split(X,y,

                                                     test_size = 0.2 , random_state = 100 )
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier 

from sklearn.metrics import accuracy_score



model_ran = RandomForestClassifier(n_estimators = 300 ).fit(train_x,train_y)

test_pred = model_ran.predict(test_x)

accuracy_score(test_y,test_pred)
model_ada = AdaBoostClassifier(n_estimators= 800).fit(train_x , train_y)

test_pred_ada = model_ada.predict(test_x)

accuracy_score(test_y,test_pred_ada)