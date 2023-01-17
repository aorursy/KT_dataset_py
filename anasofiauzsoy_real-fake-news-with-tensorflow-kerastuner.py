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
import matplotlib.pyplot as plt
real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")

fake.head()
from wordcloud import WordCloud, STOPWORDS
fake_words = ""
stopwords = set(STOPWORDS)
stopwords.add("wa")
stopwords.add("thi")
for text in fake.text.values:
    text = str(text)
    words = text.split()
    fake_words += " ".join([(i.lower() + " ") for i in words])
fake_cloud = WordCloud(width = 500, height = 500, background_color = 'white', stopwords = stopwords, min_font_size = 10)
fake_cloud.generate(fake_words)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(fake_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Fake news")
  
plt.show() 
real_words = ""
for text in real.text.values:
    text = str(text)
    words = text.split()
    real_words += " ".join([(i.lower() + " ") for i in words])
real_cloud = WordCloud(width = 500, height = 500, background_color = 'white', stopwords = stopwords, min_font_size = 10)
real_cloud.generate(real_words)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(real_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Real news")
  
plt.show() 
real["fake?"] = np.zeros(len(real))
fake["fake?"] = np.ones(len(fake))
fake.head()
from sklearn.utils import shuffle

news = real.append(fake)
news = shuffle(news)
news
import string
def clean_text(text):
    words = str(text).split()
    words = [i.lower() + " " for i in words]
    words = " ".join(words)
    words = words.translate(words.maketrans('', '', string.punctuation))
    return words

news['text'] = news['text'].apply(clean_text)
news.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(news)
train, validation = train_test_split(train, test_size = 0.2)
print(len(train), len(validation), len(test) )
train
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
trunc_type = "post"
pad_type = "post"
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train.text)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(np.array(train.text))
training_padded = pad_sequences(training_sequences,truncating=trunc_type, padding=pad_type)

max_length = len(training_padded[0])

validation_sequences = tokenizer.texts_to_sequences(np.array(validation.text))
validation_padded = pad_sequences(validation_sequences, padding=pad_type, truncating=trunc_type, maxlen = max_length)
train_x = np.copy(training_padded)
validate_x = np.copy(validation_padded)
train_y = train['fake?'].values
validate_y = validation['fake?'].values
print(len(train_x), len(train_y))
!pip install -U keras-tuner
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, hp.Int('units', min_value = 5, max_value = 200, step = 25), input_length=max_length),
        tf.keras.layers.Conv1D(16, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])), 
                      metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3)

# history = model.fit(train_x, train_y, epochs = 30, validation_data = (validate_x, validate_y),
#                    callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)])

tuner.search(train_x, train_y, epochs = 3,verbose = 2,validation_data = (validate_x, validate_y), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=2)])
tuner.results_summary()
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
#     tf.keras.layers.Conv1D(16, 5, activation='relu'),
#     tf.keras.layers.GlobalMaxPooling1D(),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(loss='binary_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(), 
#                   metrics=['accuracy'])
# history = model.fit(train_x, train_y, verbose = 2, epochs = 3, validation_data = (validate_x, validate_y),
#                    callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)])
model = tuner.get_best_models()[0]
history = model.fit(train_x, train_y, verbose = 2, epochs = 3, validation_data = (validate_x, validate_y),
                   callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=6)])
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
test_sequences = tokenizer.texts_to_sequences(np.array(test.text))
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen = max_length)
preds = np.round(model.predict(test_padded))

len(preds)
acc = np.sum(1 if i==j else 0 for i,j in zip(preds, test["fake?"].values)) / len(test)
print("Accuracy: ", acc )