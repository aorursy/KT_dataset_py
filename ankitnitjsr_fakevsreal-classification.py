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
true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
true.head()
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import Callback



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import string



from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

true["Label"] = 1

fake["Label"] = 0

data = pd.concat([true,fake],ignore_index=True)

#data_final = data[["title","Label"]]
data.head()
data['Label'].value_counts()
train , test = train_test_split(data,shuffle=True,test_size=0.33, random_state=42)
def clean(text):

    trastab = str.maketrans(string.punctuation,' '*len(string.punctuation))

    text= text.translate(trastab)

    text = text.lower()

    text = ' '.join([word for word in text.split() if word not in STOPWORDS])

    return text

    
data['title'] = data['title'].apply(clean)

data['text'] = data['text'].apply(clean)
data.head()
f = data['title'].apply(lambda x: len(x.split()))

g = data['text'].apply(lambda x: len(x.split()))
g
f.plot.hist()

g.plot.hist()
#vocab

#vc = data['text'].apply(lambda x: len(x.split()))
#hyperparameter for text and title

max_title_len = 40

max_text_len = 500

vocab = 100000

pad_type = 'post'

trunc_type = 'post'

title_embedding = 128

text_embedding = 512

embedding_dim = 512

oov_token = '<OOV>'

def preprocess(data, max_len = 40, target = 'Label', test_ratio = 0.3,depending_column = None):

    xtrain,xtest,ytrain,ytest = train_test_split(data[depending_column], data[target], test_size = test_ratio)

    tokenizer = Tokenizer(num_words = vocab,oov_token = oov_token)

    tokenizer.fit_on_texts(xtrain)

    train_sequence = tokenizer.texts_to_sequences(xtrain)

    train_padded = pad_sequences(train_sequence,maxlen = max_len, padding = pad_type,truncating = trunc_type)

    test_sequence = tokenizer.texts_to_sequences(xtest)

    test_padded = pad_sequences(test_sequence,maxlen = max_len, padding =  pad_type,truncating = trunc_type)

    return train_padded, test_padded, ytrain, ytest

    
def model_(vocab=vocab,embedding_dim = embedding_dim):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab,embedding_dim))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))

    model.add(tf.keras.layers.Dense(embedding_dim,activation='relu'))

    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['acc'])

    return model

    
def train(model, train_X, test_X, train_Y, test_Y, epochs):

    class CustomCallback(Callback):

        def on_epoch_end(self, epoch, logs={}):

            if logs.get('acc') > 0.99:

                print(f'Accuracy reached {logs.get("acc")*100:0.2f}. Stopping the training')

                self.model.stop_training = True



    history = model.fit(train_X, train_Y,

                       epochs=epochs,

                       batch_size=64,

                       validation_data=[test_X, test_Y],

                       callbacks=[CustomCallback()])

    return history
#train onm title

xtrain, xtest, ytrain, ytest = preprocess(data, max_title_len, 'Label', 0.3, 'title')

model = model_(vocab,title_embedding)

his_title = train(model,xtrain, xtest, ytrain, ytest, 15)
#train onm text

xtrain, xtest, ytrain, ytest = preprocess(data, max_title_len, 'Label', 0.3, 'text')

model = model_(vocab,text_embedding)

his_text = train(model,xtrain, xtest, ytrain, ytest, 15)
import matplotlib.pyplot as plt

title_acc = his_title.history.get('acc')

text_acc = his_text.history.get('acc')

plt.plot(title_acc, range(len(title_acc)),'r',)

#plt.plot(text_acc, range(len(text_acc)),'b')
plt.plot(text_acc, range(len(text_acc)),'b')
his_text