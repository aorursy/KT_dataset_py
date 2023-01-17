# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv').sample(10000)
# remove commonly used words and apply stemming

import nltk
stopwords = nltk.corpus.stopwords.words('english')
custom_stopwords = ['will']
stopwords.extend(custom_stopwords)

def clean_sentences(text):
    words = text.split(' ')
    clean_words = [word for word in words if word not in stopwords]
    return ' '.join(clean_words)

docs = data['question_text'].str.lower()
docs = docs.str.replace('[^a-z ]','')
docs_clean = docs.apply(clean_sentences)

path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
import gensim
embeddings = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
embeddings
#embeddings['google']
docs.index
# create vector representation for each document using word vector representation

docs_vectors = pd.DataFrame()
for doc in docs_clean:
    temp = pd.DataFrame()
    words = doc.split(' ')
    for word in words:
        try:
            word2vec = embeddings[word]
            temp = temp.append(pd.Series(word2vec),ignore_index = True)
        except:
            pass
    doc_vector = temp.mean()
    docs_vectors = docs_vectors.append(doc_vector,ignore_index = True)
docs_vectors
docs_vectors.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
docs_vectors_imputed = docs_vectors.fillna(docs_vectors.mean().mean())
docs_vectors_imputed.index
docs_vectors_imputed.index = data.index
docs_vectors_imputed.index
train , validate = train_test_split(docs_vectors_imputed,test_size = 0.3 , random_state = 300)

train_x = train
train_y = data.loc[train.index]['target']

validate_x = validate
validate_y = data.loc[validate.index]['target']

model_dt = DecisionTreeClassifier(max_depth = 20)
model_dt.fit(train_x,train_y)
validate_pred = model_dt.predict(validate_x)
print(f1_score(validate_y,validate_pred))



