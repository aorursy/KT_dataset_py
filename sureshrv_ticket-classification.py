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
import os 

import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

### Keras module importing 



from keras.models import Sequential



from keras.layers import Dense, LSTM, Dropout, Embedding,CuDNNLSTM

from keras.layers import Conv1D, MaxPooling1D



from keras.preprocessing import sequence #To convert a variable length sentence into a prespecified length



# fix random seed for reproducibility

#numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest



from gensim.models import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords



from scipy import sparse

import gensim

from scipy import sparse

from gensim.corpora import Dictionary

from gensim.models import LdaModel

import seaborn as sns

import pyLDAvis.gensim

from keras.layers import Embedding
data = pd.read_csv("../input/textclassificationfinal/all_tickets-1551435513304.csv", header=0)

logistic_data = pd.read_csv("../input/textclassificationfinal/all_tickets-1551435513304.csv", header=0)
data.shape
data = data.drop(columns=['title','ticket_type','category','sub_category1','sub_category2','business_service','impact'], axis=0)
# model = Word2Vec(body, 

#                  min_count=3,   # Ignore words that appear less than this

#                  size=200,      # Dimensionality of word embeddings

#                  workers=2,     # Number of processors (parallelisation)

#                  window=10,      # Context window for words during training

#                  iter=30) 
# y=data["urgency"]

# X=data.drop('urgency',axis=1)
# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3 , random_state = 2, stratify=y)

# X_train.shape
# cv = CountVectorizer(max_features=2000,stop_words=stopwords.words('english'), analyzer='word',token_pattern=r'\b[a-zA-Z]{3,}\b')
# X_train_data = cv.fit_transform(X_train.body)
# X_test_data = cv.fit_transform(X_test.body)
# cv.get_feature_names()
# cv.vocabulary_.items()
# full_sparse_data =  sparse.vstack([X_train_data, X_test_data])
# #Transform our sparse_data to corpus for gensim

# corpus_data_gensim = gensim.matutils.Sparse2Corpus(full_sparse_data, documents_columns=False)
# #Create dictionary for LDA model

# vocabulary_gensim = {}

# for key, val in cv.vocabulary_.items():

#     vocabulary_gensim[val] = key

    

# dict = Dictionary()

# dict.merge_with(vocabulary_gensim)
# lda = LdaModel(corpus_data_gensim, num_topics = 30 )
# data_ =  pyLDAvis.gensim.prepare(lda, corpus_data_gensim, dict)

# pyLDAvis.display(data_)
# X_train_data.shape
# X_test_data.shape
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
max_fatures = 10000

tokenizer = Tokenizer(nb_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['body'].values)

X1 = tokenizer.texts_to_sequences(data['body'].values)

X1 = pad_sequences(X1,maxlen=900)

Y1 = pd.get_dummies(data['urgency']).values

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)

print(X1_train.shape,Y1_train.shape)

print(X1_test.shape,Y1_test.shape)
embed_dim = 50

lstm_out = 100

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1], dropout=0.2))

model.add(LSTM(lstm_out, dropout_U=0.2,dropout_W=0.2))

model.add(Dense(4,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 120

model.fit(X1_train, Y1_train, nb_epoch = 3, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))

embed_dim = 200

lstm_out = 250

model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1], dropout=0.2))

model.add(CuDNNLSTM(lstm_out))

model.add(Dropout(0.3))

model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(4,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 500

model.fit(X1_train, Y1_train, nb_epoch = 3, batch_size=batch_size, verbose = 1)
score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
import matplotlib.pyplot as plt



history = model.fit(X1_train, Y1_train, validation_split=0.25, epochs=10, batch_size=500, verbose=1)



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

batch_size = 120

model.fit(X1, Y1, nb_epoch = 3, batch_size=batch_size, verbose = 2)
probability = model.predict_proba(X1)
probability_clstm_data = pd.DataFrame(probability)
probability_clstm_data.columns
for i in range(0,4):

    logistic_data[i] = proabaility_clstm_data[i]


logistic_data[0] = logistic_data[0]*1000

logistic_data[1] = logistic_data[1]*1000

logistic_data[2] = logistic_data[2]*1000

logistic_data[3] = logistic_data[3]*1000
logistic_data.describe()
logistic_data = logistic_data.drop(columns=['title','body'], axis=0)
logistic_data.dtypes
y= logistic_data["urgency"]

X= logistic_data.drop('urgency',axis=1)
X_dt_train,X_dt_test , y_dt_train,y_dt_test = train_test_split(X,y, test_size = 0.3 , random_state = 2, stratify=y)
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# scaler.fit(X_dt_train)



# X_dt_train=scaler.transform(X_dt_train)

# X_dt_test=scaler.transform(X_dt_test)

from sklearn import tree



import graphviz



import matplotlib.pyplot as plt



import math

from sklearn.metrics import accuracy_score



import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import confusion_matrix, roc_curve, auc
clf = tree.DecisionTreeClassifier(max_depth=7)

clf = clf.fit(X_dt_train,y_dt_train)
np.argsort(clf.feature_importances_)
importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

pd.DataFrame([X_dt_train.columns[indices],np.sort(importances)[::-1]])
dot_data = tree.export_graphviz(clf, out_file=None, 

                                feature_names=X_dt_train.columns,

                                class_names='target', 

                                filled=True, rounded=True, special_characters=True) 

graph = graphviz.Source(dot_data) 

graph
train_pred = clf.predict(X_dt_train )

test_pred = clf.predict(X_dt_test )
confusion_matrix_test = confusion_matrix(y_dt_test, test_pred)

confusion_matrix_train = confusion_matrix(y_dt_train, train_pred)



print(confusion_matrix_train)

print(confusion_matrix_test)
print(accuracy_score(y_dt_train,train_pred))
print(accuracy_score(y_dt_test,test_pred))