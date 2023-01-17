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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
df_news = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines = "True")
print(df_news.head)
df_news = df_news.drop("article_link", axis=1)
df_news['is_sarcastic'].value_counts().plot.bar() #equally divided
df_news.isna().sum() # check for any null entry
import string
punct = string.punctuation
def remove_punctuation(text_sentence):
    text = "".join([word for word in text_sentence if word not in punct])
    return text
df_news['headline_nopunct'] = df_news['headline'].apply(lambda x: remove_punctuation(x))
df_news.head()
    
# keras text preprocessing tokenizer is used after the pre processing but here i have used my own tokenizer using regular expression
# in preprocessing 
import re
def tokenize(text_sentence): 
    token = re.split('\W+', text_sentence)
    return token
df_news['headline_tokenize'] = df_news['headline_nopunct'].apply(lambda x: tokenize(x))
df_news.head()
import nltk
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopword(text_sentence): 
    text = [word for word in text_sentence if word not in stopwords]
    return text
df_news['headline_nostopword'] = df_news['headline_tokenize'].apply(lambda x: remove_stopword(x))
df_news.head()
ps = nltk.PorterStemmer() # i have used portstemmer but other stemmers can be used
def stemming(text_sentence): 
    text = [ps.stem(word) for word in text_sentence]
    return text
df_news['headline_stem'] = df_news['headline_nostopword'].apply(lambda x : stemming(x))
df_news.head()

# check the word veggies -> veggi , different->differ ( different and differ have very separate meaning and it wont help the neural network t
# training if it sees different and differ are same.
wm = nltk.WordNetLemmatizer()
def lemmatize(text_sentence):
    text = [wm.lemmatize(word) for word in text_sentence]
    return text
df_news['headline_lemmatize'] = df_news['headline_nostopword'].apply(lambda x : lemmatize(x))
df_news.head()

# notice that different => different is not changed.
vocab_size = 10000
embedding_dim = 16
max_len = 250
trunc_type = "post"
oov_tok = "<OOV>"
training_size = 20000
sentences = []
label = []

for i in df_news.index:
    sentences.append(df_news['headline_lemmatize'][i])
    label.append(df_news['is_sarcastic'][i])

train_sentence = sentences[0:training_size]
print("Training dataset len " , len(train_sentence))
test_sentence = sentences[training_size:]
print("Testing dataset len" , len(test_sentence))

train_label = np.array(label[0:training_size])
test_label = np.array(label[training_size:])


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentence)
word_index = tokenizer.word_index

train_word_sequence = tokenizer.texts_to_sequences(train_sentence)
train_padd_sequence = pad_sequences(train_word_sequence, maxlen=max_len, truncating=trunc_type)

test_word_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padd_sequence = pad_sequences(test_word_sequence, maxlen=max_len, truncating=trunc_type)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])
model.summary()
model.compile(optimizer="adam", loss = tf.keras.losses.binary_crossentropy, metrics = ["accuracy"])
history = model.fit(train_padd_sequence, train_label, validation_data = (test_padd_sequence, test_label), epochs = 10)
def plot_graphs(history, attr):
  plt.plot(history.history[attr])
  plt.plot(history.history['val_'+attr])
  plt.xlabel("Epochs")
  plt.ylabel(attr)
  plt.legend([attr, 'val_'+attr])
  plt.show()
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
