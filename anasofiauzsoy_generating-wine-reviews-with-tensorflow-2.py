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

from wordcloud import WordCloud, STOPWORDS
data = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

data.head()
len(data)
words = ""

stopwords = set(STOPWORDS)

for review in data.description.values:

    text = str(review)

    text = text.split()

    words += " ".join([(i.lower() + " ") for i in text])

    

cloud = WordCloud(width = 500, height = 500, background_color = 'white', stopwords = stopwords, min_font_size = 10)

cloud.generate(words)



plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(cloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("Wine Reviews Word Cloud", fontsize = 16)

    

plt.show() 
import string



def clean_text(text):

    words = str(text).split()

    words = [i.lower() + " " for i in words]

    words = " ".join(words)

    words = words.translate(words.maketrans('', '', string.punctuation))

    return words



data['description'] = data['description'].apply(clean_text)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



vocab_size = 15000

max_length = 50

oov_tok = "<OOV>"



tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(data.description.values)

word_index = tokenizer.word_index
get_word = {v: k for k, v in word_index.items()}
sequences = tokenizer.texts_to_sequences(data.description.values[::100])



n_gram_sequences = []

for sequence in sequences:

    for i,j in enumerate(sequence):

        n_gram_sequences.append(sequence[:i + 1])

        

np.array(n_gram_sequences).shape

n_gram_sequences = np.array(n_gram_sequences)

max_len = max([len(i) for i in n_gram_sequences])
padded = pad_sequences(n_gram_sequences, maxlen = max_len, padding = 'pre')

input_seq, labels = padded[:,:-1], padded[:,-1]

labels = tf.keras.utils.to_categorical(labels, num_classes = vocab_size)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten



def create_model():

    model = Sequential()

    model.add(Embedding(vocab_size, 64, input_length=max_len-1))

    #model.add(Bidirectional(LSTM(20, return_sequences = True)))

    model.add(Bidirectional(LSTM(20)))

    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

    
use_tpu = False

if use_tpu:

    # Create distribution strategy

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)



    # Create model

    with strategy.scope():

        model = create_model()

else:

    model = create_model()



model.summary()
history = model.fit(input_seq, labels, epochs=500, verbose=1)
import matplotlib.pyplot as plt



def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
review_length = int(len(words.split())/len(data))  ## average review length
seed_text = "the wine"



def write_review(seed_text):

    for _ in range(review_length):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        pred_probs = model.predict(token_list)

        predicted = np.random.choice(np.linspace(0, vocab_size - 1, vocab_size), p = pred_probs[0])

        if predicted == 1: ## if it's OOV, pick the next most likely one.

            pred_probs[0][1] = 0

            predicted = np.random.choice(np.linspace(0, vocab_size - 1, vocab_size), p = pred_probs[0])

        output_word = get_word[predicted]

        seed_text += " " + output_word

    print(seed_text)
write_review("the wine")
write_review("the wine")
write_review("the taste")
write_review("I felt")