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
dataset = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
dataset.head(20)
import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D,Conv1D,LSTM

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
dataset.isnull().values.any()
dataset.shape
dataset['review'][3]
import seaborn as sns
sns.countplot(x='sentiment', data=dataset)
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):

    return TAG_RE.sub('', text)



def preprocessing_text(text):

    # Remove html tag

    sentence = remove_tags(text)

    # Remove link

    sentence = re.sub(r'https:\/\/[a-zA-Z]*\.com',' ',sentence)

    # Remove number

    sentence = re.sub(r'\d+',' ',sentence)

    # Remove white space

    sentence = re.sub(r'\s+',' ',sentence)

    # Remove single character

    sentence = re.sub(r"\b[a-zA-Z]\b", ' ', sentence)

    # Remove bracket

    sentence = re.sub(r'\W+',' ',sentence)

    # Make sentence lowercase

    sentence = sentence.lower()

    return sentence





    
pre_proces_sen = []

sentences = list(dataset['review'])

for sen in sentences:

    pre_proces_sen.append(preprocessing_text(sen))
print(pre_proces_sen[2])
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
stop = ['has', 'its', "needn't", 'm', "wouldn't", 'but', 'he', "mustn't", 'his', 'there', 'or', "won't", 'can', 'd', "hadn't", 'how', 'hasn', 'very', 'wouldn', 'own', "doesn't", 'their', "isn't", 'an', "haven't", "wasn't", 'those', 'once', "shan't", 'when', "aren't", 've', 'it', "it's", 'of', "don't", 'and', 'down', 'yours', 'to', 'over', "she's", 'we', 'they', 'haven', 'having', 'ain', 'no', 'her', 'you', 'then', 'just', 'didn', 'into', 'before', 'shouldn', 'here', 'yourselves', 's', 'will', 'which', 'are', 'who', 'with', "you'd", 'this', 'me', 'themselves', "you've", 'hadn', 'mightn', 'she', 'o', 'more', 'whom', 'for', 'him', 'again', 'below', 'few', 'most', 'been', 'such', 'shan', 'is', 'ourselves', 'y', 'by', 'being', 'in', 'mustn', "you'll", 'herself', 'yourself', 'ours', 'between', 'had', 'other', "should've", 't', 'isn', 'them', 'himself', 're', 'doing', 'only', 'where', 'your', 'after', 'so', 'll', 'against', 'the', 'about', 'each', 'aren', 'wasn', "couldn't", 'have', 'ma', 'i', 'my', "mightn't", 'as', 'from', 'itself', 'under', 'same', 'why', 'any', 'our', 'be', 'off', "hasn't", 'through', "you're", 'was', 'did', "shouldn't", 'myself', 'some', 'theirs', 'hers', 'further', 'do', 'now', 'than', 'too', 'during', 'at', 'because', 'doesn', 'needn', "weren't", 'don', "didn't", 'couldn', 'what', 'does', 'if', 'up', 'on', 'these', 'should', 'all', "that'll", 'above', 'weren', 'that', 'a', 'while', 'both', 'until', 'were', 'am']
for i in range(len(pre_proces_sen)):

    x = pre_proces_sen[i]

    x = word_tokenize(x)

    new_x_list = [word for word in x if word not in stop]

    pre_proces_sen[i] = ' '.join(new_x_list)

    if i% 2000 == 0:

        print(i,end=" ")

print(pre_proces_sen[2])
y  = dataset['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
X = pre_proces_sen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
import gensim

WORD2VEC_MODEL = "../input/one-lac-bin/model_1_lac.bin"

#load word2vec model

word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True, limit=100000)
embedding_weights = np.zeros((vocab_size, 300))

for word, index in tokenizer.word_index.items():

    #embedding_vector = word2vec.get(word)

    try:

        embedding_weights[index] = word2vec[word]

    except:

        pass 
print(word2vec['not'][:40])
model = Sequential()



embedding_layer = Embedding(vocab_size, 300, weights=[embedding_weights], input_length=maxlen , trainable=False)

model.add(embedding_layer)



model.add(Conv1D(128, 5, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.2)),

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)



score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])

print("Test Accuracy:", score[1])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()
model = Sequential()

embedding_layer = Embedding(vocab_size, 300, weights=[embedding_weights], input_length=maxlen , trainable=False)

model.add(embedding_layer)

model.add(LSTM(128))



model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history =  model.fit(X_train, y_train,  batch_size=128, epochs=8, validation_split=0.2,verbose=1)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])

print("Test Accuracy:", score[1])
import matplotlib.pyplot as plt



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()
text = ['I laughed all the way through this rotten movie It so unbelievable woman leaves her husband after many years of marriage has breakdown in front of real estate office What happens The office manager comes outside and offers her job Hilarious Next thing you know the two women are going at it Yep they re lesbians Nothing rings true in this Lifetime for Women with nothing better to do movie Clunky dialogue like don want to spend the rest of my life feeling like had chance to be happy and didn take it doesn help There a wealthy distant mother who disapproves of her daughter new relationship sassy black maid unbelievable that in the year film gets made in which there a sassy black maid Hattie McDaniel must be turning in her grave The woman has husband who freaks out and wants custody of the snotty teenage kids Sheesh No cliche is left unturned']
pre = text

pre_sequences = tokenizer.texts_to_sequences(pre)

pre_padded = pad_sequences(pre_sequences,maxlen=maxlen, padding='post')

prediction = model.predict(pre_padded)

prediction