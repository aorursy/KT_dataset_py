# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import json
from sklearn.cross_validation import train_test_split
# Any results you write to the current directory are saved as output.
import gensim
# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

def getSenses(data):
    text_data= []
    for doc in data:
        text_data.append(doc['ingredients'])
    return text_data 


train_text = generate_text(train)
test_text = generate_text(test)
Y = [doc['cuisine'] for doc in train]
_Id = [doc['id'] for doc in test]
train_sentences = getSenses(train)
test_sentences = getSenses(test)
sentences = train_sentences + test_sentences
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(
    max_vocab_size=10**5*2,
    min_count=5,
    size=200,
)
model.build_vocab(sentences=sentences)
model.train(sentences=sentences, epochs=500, total_examples=model.corpus_count)
model.save('word2vec')
# modelFT = gensim.models.FastText(size=200, window=7, min_count=1, iter=100)

# # build the vocabulary
# modelFT.build_vocab(sentences)

# # train the model
# modelFT.train(sentences, epochs=500, total_examples=modelFT.corpus_count)

# print(modelFT)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(vocabulary=model.wv.index2word, max_df=0.9, min_df=2)
model.wv.index2word
X = tfidf.fit_transform((' '.join(i) for i in sentences))
from scipy.sparse import csr_matrix, hstack, vstack
X_train_source = tfidf.transform((' '.join(i) for i in train_sentences))
X_test_source = tfidf.transform((' '.join(i) for i in test_sentences))
X_train_tfidf,  X_test_tfidf, Y_train, Y_test = train_test_split(X_train_source, Y, test_size=0.2)
X_train_w2v = X_train_tfidf.dot(model.wv.vectors)
X_test_w2v = X_test_tfidf.dot(model.wv.vectors)
X_train_w2v.shape
X_train = hstack((X_train_tfidf, X_train_w2v))
X_test = hstack((X_test_tfidf, X_test_w2v))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
lg = LogisticRegression(
    penalty='l2',
    C=100, 
    n_jobs=-1, verbose=1, 
    
    solver='sag', multi_class='multinomial',
    
    max_iter=500
)
lg.fit(X_train, Y_train)
i = 0
w2vwocab = []
for item in range(200):
    w2vwocab.append('w2v' + str(item))
    i=i+1
    

w2vocabDict = dict.fromkeys(w2vwocab, 0)
w2vocabDict
# Достаём из векторайзера словарь
# vocab = {**tfidf.vocabulary_, **w2vocabDict}
# vocab = vocab.items()
# vocab = sorted(list(vocab), key=lambda x: x[1])
# vocab_words, vocab_index = zip(*vocab)
# vocab_words = np.array(vocab_words)
# for label in range(20):
#     _class_coef = lg.coef_[label]
#     print('Class', label, 'слова увеличивающие вероятность класса:')
#     print(list(vocab_words[ (-_class_coef).argsort()][:100]))
#     print()
#     print('Class', label,  'слова уменьшающие вероятность класса:')
#     print(list(vocab_words[ (_class_coef).argsort()][:100]))
#     print('-'*80)
Y_pred = lg.predict(X_test)
print(classification_report(Y_test, Y_pred, digits=6))
score = lg.score(X_test, Y_test)
print(score)
X_ = hstack((X_test_source, X_test_source.dot(model.wv.syn0)))
Y_target = lg.predict(X_)
with open('tfidf_w2v_lg2.csv', 'w') as f:
    f.write('id,cuisine\n')
    for _id, y  in zip(_Id, Y_target):
        f.write('%s,%s\n' % (_id, y))
from sklearn.neural_network import MLPClassifier
clfMLP = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(1000,), random_state=1)
clfMLP.fit(X_train, Y_train)                         

score = clfMLP.score(X_test_tfidf, Y_test)
print(score)
Y_target = clfMLP.predict(X_test_source)
with open('tfidf_w2v_mlp2.csv', 'w') as f:
    f.write('id,cuisine\n')
    for _id, y  in zip(_Id, Y_target):
        f.write('%s,%s\n' % (_id, y))
