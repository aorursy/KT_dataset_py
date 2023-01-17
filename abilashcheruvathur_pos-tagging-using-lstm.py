import tensorflow
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Initialize the random number generator
import random
random.seed(0)

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")
# IMPORT DATA
import pandas as pd
import numpy as np

data = pd.read_csv('/kaggle/input/ner_dataset-1.csv', encoding='latin1')
data = data.fillna(method="ffill") # Deal with N/A
data.head()
data.shape
data.info()
tags = list(set(data["POS"].values)) # Read POS values
tags # List of possible POS values
words = list(set(data["Word"].values))
words.append("DUMMY") # Add a dummy word to pad sentences.
words
# Code to read sentences

class ReadSentences(object): 
    
    def __init__(self, data):
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
sentences = ReadSentences(data).sentences # Read all sentences
sentences[1]
# Convert words and tags into numbers
word2id = {w: i for i, w in enumerate(words)}
tag2id = {t: i for i, t in enumerate(tags)}
word2id
tag2id
# Prepare input and output data

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 50
X = [[word2id[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=len(words)-1)
y = [[tag2id[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2id["."])
# Convert output to one-hot bit

from tensorflow.keras.utils import to_categorical
y = [to_categorical(i, num_classes=len(tags)) for i in y]
y[0]
X[0]
# Training and test split by sentences

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input
input = Input(shape=(max_len,)) # Input layer
model = Embedding(input_dim=len(words), output_dim=50, input_length=max_len)(input) # Word embedding layer
model = Dropout(0.1)(model) # Dropout
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model) # Bi-directional LSTM layer
out = TimeDistributed(Dense(len(tags), activation="softmax"))(model)  # softmax output layer
model = Model(input, out) # Complete model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]) # Compile with an optimizer
history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=3, validation_split=0.1, verbose=1) # Train
# Demo test on one sample. See how it is mostly correct, but not 100%

i = 1213 # Some test sentence sample
p = model.predict(np.array([X_te[i]])) # Predict on it
p = np.argmax(p, axis=-1) # Map softmax back to a POS index
for w, pred in zip(X_te[i], p[0]): # for every word in the sentence
    print("{:20} -- {}".format(words[w], tags[pred])) # Print word and tag
import nltk
nltk.download('punkt')
from nltk import word_tokenize

sentence = nltk.word_tokenize('That was a nice jump')
X_Samp = pad_sequences(maxlen=max_len, sequences=[[word2id[word] for word in sentence]], padding="post", value=len(words)-1)
p = model.predict(np.array([X_Samp[0]])) # Predict on it
p = np.argmax(p, axis=-1) # Map softmax back to a POS index
for w, pred in zip(X_Samp[0], p[0]): # for every word in the sentence
    print("{:20} -- {}".format(words[w], tags[pred])) # Print word and tag
