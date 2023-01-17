# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/amazonreviews/outpu.xls')

df=df.dropna()
df
import nltk
import string
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    
    return text
# apply the above function to df['text']
df['comment'] = df['comment'].map(lambda x: clean_text(x))
embeddings_index = dict()
f = open('/kaggle/input/glovetwitter100d/glove.twitter.27B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
vocab=20000
tokenizer=Tokenizer(vocab,oov_token="<oov>")
tokenizer.fit_on_texts(df['comment'])
sequence=tokenizer.texts_to_sequences(df['comment'])
padded=pad_sequences(sequence,maxlen=50)
embedding_matrix = np.zeros((vocab, 100))
for word, index in tokenizer.word_index.items():
    if index > vocab - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
padded.shape
y=df.stars
y.value_counts()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y=le.fit_transform(y)
print(y)
from keras.utils import to_categorical
cat=to_categorical(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(padded,cat,test_size=0.1,stratify=y,random_state=42)
print(X_train.shape)
print(y_train.shape)
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,GlobalAveragePooling1D,Dropout,Conv1D,MaxPooling1D,GRU
from keras.optimizers import Adam,SGD
model=Sequential()
model.add(Embedding(vocab, 100, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.2))
# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(250,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(GRU(100))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer=Adam(lr=0.0002),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train, batch_size=32,validation_split=0.1,epochs=4)

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
import matplotlib.pyplot as plt
import seaborn as sns
plt.title('Loss')
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='test')
plt.legend()
plt.show();
plt.title('Accuracy')
plt.plot(model.history.history['accuracy'], label='train')
plt.plot(model.history.history['val_accuracy'], label='test')
plt.legend()
plt.show();

new_complaint = ['poor quality dont buy waste of money']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=50)
pred = model.predict(padded)
labels = ['1', '2', '3', '4', '5']
print(seq,pred, labels[np.argmax(pred)])
new_complaint = ['can buy if you want to']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=50)
pred = model.predict(padded)
labels = ['1', '2', '3', '4', '5']
print(seq,pred, labels[np.argmax(pred)])
new_complaint = ['Loved it thankyou amazon']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=50)
pred = model.predict(padded)
labels = ['1', '2', '3', '4', '5']
print(seq,pred, labels[np.argmax(pred)])
y_pred=model.predict_classes(X_test)
y_test=y_test.argmax(axis=1)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
