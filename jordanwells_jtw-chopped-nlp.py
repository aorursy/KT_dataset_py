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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
chopped_data = pd.read_csv("/kaggle/input/chopped-10-years-of-episode-data/chopped.csv")
chopped_data.head()
round_type = "entree"
round_list = chopped_data[round_type].to_list()
print(round_list[:5])
text = []

for selection in round_list:                     # For each basket selection in appetizers
    ingredients_comma =  selection.split(",")         # Convert each basket selection into a list of strings
    ingredients = []
    for ingredient in ingredients_comma:
        for element in ingredient.strip().split(" "):
            ingredients.append(element)
    for i in range(1, len(ingredients)):              # For as long as the list of strings is
        n_gram = ingredients[:i + 1]                  # create a fragment of the sentence
        text.append(n_gram)                           # and append it to text

print(text[:15])
tokenizer = Tokenizer(num_words = 2000, oov_token = "<OOV>")    # Generate a tokenizer
tokenizer.fit_on_texts(text)                                    # and fit it on our text
sequences = tokenizer.texts_to_sequences(text)                  # Turn all of our text into texts
word_index = tokenizer.word_index                               
word_count = len(word_index) - 1
max_len = max([len(sequence) for sequence in sequences])
sequences = np.array(pad_sequences(sequences, maxlen = max_len, padding = "pre"))
print(sequences)
xs = sequences[:,:-1]
labels = sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(word_count, 64, input_length = max_len - 1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(word_count +2, activation = "softmax"),
])

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(xs, ys, epochs=300, verbose = 2)
seed_ingredients = "chicken"
num_words = 6

for _ in range(num_words):
    token_list = tokenizer.texts_to_sequences([seed_ingredients])[0]
    token_list = pad_sequences([token_list], maxlen = max_len - 1, padding = "pre")
    predicted = model.predict_classes(token_list, verbose = 2)
    output_word = ""
    for word, i in tokenizer.word_index.items():
        if i == predicted:
            output_word = word
            break
    seed_ingredients += f" {output_word}"

print(seed_ingredients)