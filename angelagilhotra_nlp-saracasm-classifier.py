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
import json



def parse_data(file):

    for l in open(file,'r'):

        yield json.loads(l)



data = list(parse_data('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json'))



sentences = []

labels = []

urls = []



for item in data:

    sentences.append(item['headline'])

    labels.append(item['is_sarcastic'])

    urls.append(item['article_link'])

    

# sentences
import tensorflow as tf

from tensorflow import keras



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 10000

embedding_dim = 16

max_length = 100

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

training_size = 20000
training_sentences = sentences[0:training_size]

testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]

testing_labels = labels[training_size:]
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index



training_sequences = tokenizer.texts_to_sequences(training_sentences)



training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating = trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
## classifying on Sentiment



model = keras.Sequential([

    # embedding = direction of each word is learned epoch by epoch

    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    # adding up the vectors

    # keras.layers.GlobalAveragePooling1D(),

    keras.layers.Bidirectional(keras.layers.LSTM(64)),

#     keras.layers.Bidirectional(keras.layers.LSTM(64), return_sequences=True),

    # keras.layers.Dense(24, activation='relu'),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
import numpy as np

training_padded = np.array(training_padded)

training_labels = np.array(training_labels)

testing_padded = np.array(testing_padded)

testing_labels = np.array(testing_labels)





num_epochs = 30



history = model.fit(

    training_padded, 

    training_labels,

    epochs=num_epochs, 

    validation_data=(

        testing_padded, testing_labels

    ), 

    verbose=1

)

sentence = [

    "granny starting to fear spiders in the garden might be real",

    "the weather today is bright and sunny",

    "How I accidently grew a melon the size of a buffalo",

    "Biden Campaign Whittles VP Shortlist Down To Either Woman Or Man With Long Hair",

    "Theoretical Astro-Fetishists Posit Black Holes Could Be Used For Anonymous Sex Across Parallel Universes",

#     "Facebook Announces Plan To Break Up U.S. Government Before It Becomes Too Powerful",

    "Facebook announces new application",

    "Facebook",

    "A melon grew the size of a buffalo",

    "One day I will crush you",

    "My Advice To Anyone Starting A Business Is To Remember That Someday I Will Crush You"

]





## Predictions on RNN



sequences = tokenizer.texts_to_sequences(sentence)

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

predictions = model.predict(padded)



# print(predictions * 10)

result = []

for i in range (len(sentence)):

    result.append((sentence[i], predictions[i]))

    

result
## classifying on Sentiment



model2 = keras.Sequential([

    # embedding = direction of each word is learned epoch by epoch

    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    # adding up the vectors

    keras.layers.GlobalAveragePooling1D(),

    keras.layers.Dense(24, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])



model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.summary()
num_epochs = 30



history2 = model2.fit(

    training_padded, 

    training_labels,

    epochs=num_epochs, 

    validation_data=(

        testing_padded, testing_labels

    ), 

    verbose=1

)

sentence2 = [

    "granny starting to fear spiders in the garden might be real",

    "the weather today is bright and sunny",

    "How I accidently grew a melon the size of a buffalo",

    "Biden Campaign Whittles VP Shortlist Down To Either Woman Or Man With Long Hair",

    "Theoretical Astro-Fetishists Posit Black Holes Could Be Used For Anonymous Sex Across Parallel Universes",

#     "Facebook Announces Plan To Break Up U.S. Government Before It Becomes Too Powerful",

    "Facebook announces new application",

    "Facebook",

    "A melon grew the size of a buffalo",

    "One day I will crush you",

    "My Advice To Anyone Starting A Business Is To Remember That Someday I Will Crush You"

]







sequences2 = tokenizer.texts_to_sequences(sentence2)

padded2 = pad_sequences(sequences2, maxlen=max_length, padding=padding_type, truncating=trunc_type)

predictions2 = model2.predict(padded2)



# print(predictions * 10)

result2 = []

for i in range (len(sentence2)):

    result2.append((sentence2[i], predictions2[i]))

    

result2