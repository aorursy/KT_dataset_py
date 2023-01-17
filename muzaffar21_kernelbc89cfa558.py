# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("."))
print("hello")

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
import nltk as nl


%matplotlib inline


data = pd.read_csv("../input/train.csv")
print(data.info())
print(data.target.value_counts())
train_text = data['question_text'].values[:50]
train_target = data['target'].values[:50]
test_text = data['question_text'].values[50:60]
test_target = data['target'].values[50:60]

# here I am just taking small chunk of data to commit fast
def get_one_hot_y(target):
    y =[]
    for t in target:
        z = np.zeros(2, dtype=float)
        z[t] = 1
        y.append(z)
    y = np.array(y)
    return y

from gensim.models import KeyedVectors
text_model = KeyedVectors.load_word2vec_format('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
print("done loading word vec model ")

print (text_model.most_similar('desk'))
def sent_vectorizer(sentence, text_model):
    words = nl.word_tokenize(sentence)
    sent_vec = np.mean([text_model[w] for w in words if w in text_model]
                    or [np.zeros(300)], axis=0)
    return sent_vec
    
def text_vectorizer(text, text_model):
    X = np.zeros((len(text),300),dtype=float)
    for i,s in enumerate(text):
        if i% 1000 == 0:
            pass
            #print (i)
        vec = sent_vectorizer(s, text_model)
        #print (vec.shape)
        X[i] = vec
    return X
max_words = 1000
max_len = 300
X = text_vectorizer(train_text, text_model)
print( X.shape)

print("done....")
y = get_one_hot_y(train_target)
print("done..")
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
print("done ......")
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(X,y,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
X_test = text_vectorizer(test_text,text_model)
y_test = get_one_hot_y(test_target)
accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

submit_data = pd.read_csv("../input/test.csv")
submit_data.info()
submit_text = submit_data["question_text"]
X_submit = text_vectorizer(submit_text,text_model)
print("done ....")
prediction = model.predict(X_submit)
print("done ..")
p = np.argmax(prediction, axis =1)
print (p)
my_submission = pd.DataFrame( {'qid': submit_data.qid , 'prediction': p } )
my_submission.to_csv('submission.csv', index=False)
print("done....")

print(os.listdir("."))

l = [line for line in open('submission.csv')]
