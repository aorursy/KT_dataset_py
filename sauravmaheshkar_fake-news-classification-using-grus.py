import numpy as np # For Linear Algebra

import pandas as pd # For I/O, Data Transformation

import tensorflow as tf # Tensorflow

import tensorflow_datasets as tfds # For the SubTextEncoder

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fakedataset = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv") # Make a DataFrame for Fake News

realdataset = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv") # Make a DataFrame for Real News

realdataset["class"] = 1 # Adding Class to Real News

fakedataset["class"] = 0 # Adding Class to Fake News

realdataset["text"] = realdataset["title"] + " " + realdataset["text"] # Concatenating Text and Title into a single column for Real News DataFrame

fakedataset["text"] = fakedataset["title"] + " " + fakedataset["text"] # Concatenating Text and Title into a single column for Fake News DataFrame

realdataset = realdataset.drop(["subject", "date", "title"], axis = 1) # Removing Redundant features from Real News DataFrame

fakedataset = fakedataset.drop(["subject", "date", "title"], axis = 1) # Removing Redundant features from Fake News DataFrame

dataset = realdataset.append(fakedataset, ignore_index = True) # Making a Single DataFrame 

del realdataset, fakedataset 
vocab_size = 10000

encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(dataset["text"], vocab_size)
def enc(dataframe):

    tokenized = []

    for sentence in dataframe["text"].values:

        tokenized.append(encoder.encode(sentence))

    out = tf.keras.preprocessing.sequence.pad_sequences(tokenized, padding = "post")

    return out

x = enc(dataset)
y = dataset["class"]

print(y)
# Model Definition

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(encoder.vocab_size, 64), # Embedding Layer using the vocab-size from encoder

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64,  return_sequences=True)), # Create the first Bidirectional layer with 64 LSTM units

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)), # Second Bidirectional layer witth 32 LSTM units

    tf.keras.layers.Dense(64, activation='relu'), # A Dense Layer with 64 units

    tf.keras.layers.Dropout(0.5), # 50% Dropout

    tf.keras.layers.Dense(1) # Final Dense layer with a single unit

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['acc']) # Compiling the Model
history = model.fit(x,y, epochs = 2)
def pad_to_size(vec, size):

  zero = [0] * (size - len(vec))

  vec.extend(zeros)

  return vec



def sample_predict(sample_pred_text, pad):

  encoded_sample_pred_text = encoder.encode(sample_pred_text)



  if pad:

    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)

  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)

  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))



  return (predictions)



sample_pred_text = ('The movie was cool. The animation and the graphics')

predictions = sample_predict(sample_pred_text, pad=False)

print(predictions)
model.save('my_model.h5') 

import os

from IPython.display import FileLink

FileLink(r'my_model.h5')