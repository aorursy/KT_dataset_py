# Import libraries

import pandas as pd

import numpy as np

from tensorflow import keras

from keras.models import model_from_json, Sequential

import os

from sklearn.model_selection import train_test_split

import csv

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
# Load our amazon and gibberish data. The gibberish data has a column for the response and the label (0 for gibberish)

gibberish = pd.read_csv("../input/gibberish-text-classification/Gibberish.csv", encoding = "ISO-8859-1")

amazon = pd.read_csv("../input/gibberish-text-classification/Amazon.csv", encoding = "ISO-8859-1")
# Take a peak at the Amazon data

gibberish.head()
# Take a peak at the Amazon data

amazon.head()
# Drop the first column since we don't need it, then rename the remaining column to Response to match the gibberish data

# Then add a column for Label and set the values to 0 since none of them are gibberish

amazon.drop(amazon.columns[0], inplace=True, axis=1)

amazon.columns = ["Response"]

amazon["Label"] = 0

amazon.head()
# Take a look at the length of each dataset

len(gibberish), len(amazon)
# Drop the majority of the Amazon reviews to reduce training time and to downsample

drop_indices = np.random.choice(amazon.index, 800000, replace=False)

amazon = amazon.drop(drop_indices)
# Define a function to remove the intro to the review using the : to identity the review title

def remove_intro(x):

  if x.find(":") < 0:

    return x

  else:

    return x[x.find(":") + 1:len(x)].strip()
# Remove the intro to the reviews to make it flow more like natural language

amazon["Response"] = amazon["Response"].apply(remove_intro)
# Combine the Amazon and gibberish data

X = np.concatenate((amazon["Response"].values, gibberish["Response"].values))

Y = np.concatenate((amazon["Label"].values, gibberish["Label"].values))
# Make sure our features and labels are the same length

len(X), len(Y)
# Review the mean length of the reviews vs the gibberish

pd.DataFrame({"Response": X, "Label" : Y} ).groupby("Label")["Response"].apply(lambda x: np.mean(x.str.len()))
# Review length of all the responses

response_word_count = []



for i in range(len(X)):

  response_word_count.append(len(str(X[i])))

  

length_df = pd.DataFrame({"Response": response_word_count})

length_df.hist(bins=30)



plt.show()

np.percentile(response_word_count, 80), np.median(response_word_count)
# after some trial and error I found a max length around 250 produced the best results

max_len = 256
''''

Create a tokenizer to convert the text into integer sequences. For this problem, we want to convert each character

to an integer so we set char_level to True.

'''

tokenizer = Tokenizer(char_level=True)

tokenizer.fit_on_texts(X)
# Review all the characters found in our text

tokenizer.word_index
# Use our tokenizer to convert our text features to integer values that can be fed to the Embedding layer

X = tokenizer.texts_to_sequences(X)
# Take a look at a review to confirm it has been cnverted to integers

X[1]
# We need to pad our integer sequences so that our reviews are all the same length when we feed them to our Embedding layer

X = pad_sequences(X, maxlen=max_len, padding="post")
# Split our data into training and test sets

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.10)
# set our vocabulary size

vocab_size = len(tokenizer.word_index) + 1
# Build and compile our model

model = keras.models.Sequential([

    keras.layers.Embedding(vocab_size, 64),

    keras.layers.GlobalAveragePooling1D(),

    keras.layers.Dense(32, activation="relu"),

    keras.layers.Dropout(rate=0.25),

    keras.layers.Dense(2, activation="softmax")

])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
# Train our model

history = model.fit(np.array(x_train), y_train, epochs=5, validation_data=(np.array(x_test), y_test), verbose=1)
# Lets plot the accuracy to check for overfitting

loss_train = history.history['acc']

loss_val = history.history['val_acc']

epochs = range(1,6)

plt.plot(epochs, loss_train, 'g', label='Training accuracy')

plt.plot(epochs, loss_val, 'b', label='Validation accuracy')

plt.title('Training And Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# Define a funtion to test out some made up reviews/gibberish

def predict_result(text):

    result = {0:"REAL REVIEW", 1: "GIBBERISH"}

    print(result[model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_len, padding="post"))[0]])
predict_result("OMG - This product is so great! Can't wait to purchase again!")
predict_result("agkvjajehijioore jkafdghahjhjh")
predict_result("askdglreegiyyyyyyyy")
predict_result("Great quality and customer service, I will come back in the future...")
predict_result("Fast delivery and it works perfect. Highly recommended")