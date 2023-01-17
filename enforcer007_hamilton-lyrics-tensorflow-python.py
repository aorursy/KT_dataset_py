import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Import Packages



import tensorflow as tf

from pathlib import Path

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction import text

import matplotlib.pyplot as plt

import string

import re
input_folder = Path("/kaggle/input")

input_file = input_folder/"hamilton-lyrics"/"ham_lyrics.csv"
df = pd.read_csv(input_file,encoding = "ISO-8859-1")
df.head()
# Get Title Counts

df['title'].value_counts().sort_index()
# Filter Data with more than 3 words

df = df[df['lines'].apply(lambda x: len(x.split(" ")) > 3)]
# Punctuation Regex

punct = re.compile(r'[!\\"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~0-9]+')
#Get Frequency Counts after processing => Lowercase + remove numbers, punctuation + strip whitespace

cv = text.CountVectorizer(lowercase=True,preprocessor=lambda x: punct.sub("",x.strip()).lower(),stop_words='english')
op = cv.fit_transform(df["lines"])
df_freq = pd.DataFrame(op.toarray(),columns=cv.get_feature_names())
df_freq.head()
freq_words = df_freq.sum(axis=0)
freq_words.sort_values(ascending=False)
wc = WordCloud(width=600,height=300).generate_from_frequencies(freq_words)
plt.rcParams["figure.figsize"] = (20,5)

plt.imshow(wc)
#Store processed text in a new column

df['cleaned_lines'] = df['lines'].apply(lambda x: punct.sub("",x.strip()).lower())
# Join lines of a song by title

df_song = df.groupby('title',sort=False).apply(lambda x: " ".join(x['cleaned_lines']))
df_song.iloc[0]
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
num_words = 5000

oov_token = '<UNK>'

pad_type = 'post'

trunc_type = 'post'
tokenizer = Tokenizer(num_words=num_words,oov_token=oov_token)
tokenizer.fit_on_texts(df_song)
seqs = tokenizer.texts_to_sequences(df_song)
n_grams = 11

gram_seqs = []

n_seqs = len(seqs)

for i in seqs:

    n_i = len(i)

    for j in range(n_i-n_grams):

        gram_seqs.append(i[j:j+n_grams])
labels = [i[-1] for i in gram_seqs]

inputs = [i[:-1] for i in gram_seqs]
from sklearn.preprocessing import OneHotEncoder

from keras.utils import to_categorical

from keras import Model

from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional
encoded_labels = to_categorical(labels,num_classes=num_words)
class lyrics_generator(Model):

    def __init__(self):

        super(lyrics_generator,self).__init__()

        self.embedding = Embedding(num_words,64,input_length=n_grams-1)

        self.lstm = Bidirectional(LSTM(20))

        self.dense = Dense(num_words,activation='softmax')

    

    def call(self,x):

        x = self.embedding(x)

        x = self.lstm(x)

        x = self.dense(x)

        return x

    

    def model(self):

        x = Input(shape=(n_grams-1))

        return Model(inputs=[x], outputs=self.call(x))
m = lyrics_generator()
dataset = tf.data.Dataset.from_tensor_slices((inputs,encoded_labels)).batch(64)
m.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

          ,loss=tf.keras.losses.CategoricalCrossentropy()

         ,metrics=[tf.keras.metrics.CategoricalAccuracy()])
m.model().summary()
history = m.fit(dataset,epochs=200,verbose=0)
print("Loss: {} and Accuracy: {}".format(history.history['loss'][-1],history.history['categorical_accuracy'][-1]))
def write_lyric(text,text_length=10):

    for i in range(text_length):

        seqs_test = tokenizer.texts_to_sequences([text])

        seqs_test = pad_sequences(seqs_test,maxlen=n_grams-1,value=1)

        pred_probs = m(seqs_test)

        index = tf.argmax(pred_probs,axis=1)[0].numpy()

        word = tokenizer.index_word[index]

        text = text+" "+word

    return text
write_lyric("the man")
write_lyric("he was")
write_lyric("alexander")
write_lyric("there was")
write_lyric("it has")
write_lyric("I am")
write_lyric("Eliza")
write_lyric("sir")
write_lyric("Thomas Jefferson",text_length=50)