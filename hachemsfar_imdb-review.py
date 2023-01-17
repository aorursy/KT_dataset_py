from importlib import reload

import sys

from imp import reload

import warnings

warnings.filterwarnings('ignore')

if sys.version[0] == '2':

    reload(sys)

    sys.setdefaultencoding("utf-8")
import time



montemps=time.time()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd



df1 = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter="\t")

df1 = df1.drop(['id'], axis=1)

df1.head()
df2 = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")

df2.head()
df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)

df2.columns = ["review","sentiment"]

df2.head()
df2 = df2[df2.sentiment != 'unsup']

df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})

df2.head()
df = pd.concat([df1, df2]).reset_index(drop=True)

df.head()
import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()





def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
df.head()
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
loss_dict={}



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers

import seaborn as sns

import matplotlib.pyplot as plt #for plotting

import keras

import pickle

%matplotlib inline



max_features = 6000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(df['Processed_Reviews'])

list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])



maxlen = 130

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

y = df['sentiment']
loss_dict={}
df1 = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter="\t")

df1 = df1.drop(['id'], axis=1)

df2 = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")

df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)

df2.columns = ["review","sentiment"]

df2 = df2[df2.sentiment != 'unsup']

df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})

df = pd.concat([df1, df2]).reset_index(drop=True)

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))

df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()

max_features = 6000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(df['Processed_Reviews'])

list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])
loss_dict={}




for i in [keras.optimizers.SGD(),keras.optimizers.adam(),keras.optimizers.Adamax()]:

    montemps=time.time()

    maxlen = 130

    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

    y = df['sentiment']    

    

    embed_size = 128

    model = Sequential()

    model.add(Embedding(max_features, embed_size))

    model.add(Bidirectional(LSTM(32, return_sequences = True)))

    model.add(GlobalMaxPool1D())

    model.add(Dense(20, activation="relu"))

    model.add(Dropout(0.05))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer=i, metrics=['accuracy'])



    batch_size = 2048

    epochs = 32

    h =model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    

    print(h.history.keys())

    accuracy = h.history['acc']

    val_accuracy = h.history['val_acc']

    loss = h.history['loss']

    val_loss = h.history['val_loss']

    epochs = range(len(accuracy))

    loss_dict[str(str(i).split("keras.optimizers.")[1].split("()")[0])]=loss

    loss_dict[str(str(i).split("keras.optimizers.")[1].split("()")[0])+"acc"]=h.history['acc']

    y=time.time()-montemps

    print(i)

    print(y)


print(loss_dict)
f = open("loss_dict.pkl","wb")

pickle.dump(loss_dict,f)

f.close()
epochs = range(len(loss_dict['SGD object at 0x7f85339c9908>']))
plt.plot(epochs, loss_dict['SGD object at 0x7f85339c9908>'], 'b', label='SGD')

plt.plot(epochs, loss_dict['Adam object at 0x7f85339a3f28>'], 'r', label='ADAM')

plt.plot(epochs, loss_dict['Adamax object at 0x7f85339c9978>'], 'g', label='ADAMAX')



plt.title('Total Loss')

plt.legend()

plt.show()

plt.figure()