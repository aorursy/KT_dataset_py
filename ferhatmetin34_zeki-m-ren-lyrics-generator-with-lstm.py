# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import tensorflow as tf

from tensorflow.keras.layers import GRU, LSTM, Embedding

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import optimizers

from tensorflow.keras.layers import Activation, Dense, Bidirectional

import nltk

from textblob import Word

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional,BatchNormalization

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers

import tensorflow as tf

from tensorflow import keras

import tensorflow.keras.utils as ku 

import numpy as np 

import os

data=pd.read_csv("/kaggle/input/zeki-mren-ark-szleri/zeki.csv")

df=data.copy()
df.head()
df=df.drop("index",axis=1)
df.head()
df["lyric"]=df["lyric"].str.lstrip("[").str.rstrip("]")
df.head()
df['lyric'] = df['lyric'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df['name'] = df['name'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df.head()
df['lyric'] = df['lyric'].str.replace('[^\w\s]','')

df['lyric'] = df['lyric'].str.replace('\d','')

df['name'] = df['name'].str.replace('\d',' ')
df.head()
df["lyric"][2]
df[df["lyric"]=="şarkı enstrümantal olduğu için şarkı sözü bulunmamaktadır "]
df=df.drop(df[df["lyric"]=="şarkı enstrümantal olduğu için şarkı sözü bulunmamaktadır "].index)
df.shape
df.head()
plt.figure(figsize=(15,6))

pd.Series(' '.join(df['lyric']).split()).value_counts().sort_values()[-50:].plot.bar(color="r")

plt.grid()
df.head()
tf1 = (df["lyric"]).apply(lambda x: 

                             pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ["words","tf"]
tf1[tf1.tf>100]
with open("zeki_muren.txt","w") as file:

    for i in df.lyric:

        file.write(i +'\n')
data_txt = open('zeki_muren.txt').read()
data_txt[:100]
tf1[tf1.tf>100].sort_values(by="tf").plot.bar(x = "words", y = "tf");
tokenizer = Tokenizer()

corpus = data_txt.lower().split("\n")





tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1



# create input sequences using list of tokens

input_sequences = []

for line in corpus:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i+1]

        input_sequences.append(n_gram_sequence)
corpus
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))



# create predictors and label

predictors, label = input_sequences[:,:-1],input_sequences[:,-1]



label = ku.to_categorical(label, num_classes=total_words)







model = Sequential()

model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))

model.add(Bidirectional(LSTM(200, return_sequences = True)))

model.add(Dropout(0.2))

model.add(LSTM(100))

model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.05)))

model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())



history = model.fit(predictors, label, epochs=150, verbose=1)
acc = history.history['accuracy']

loss = history.history['loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.title('Training accuracy')
plt.plot(epochs, loss, 'b', label='Training Loss')

plt.title('Training loss')

plt.legend()



plt.show()



seed_text = "gitme sana muhtacım"

next_words = 100

  

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    predicted = model.predict_classes(token_list, verbose=0)

    output_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted:

            output_word = word

            break

    seed_text += " " + output_word

print(seed_text)