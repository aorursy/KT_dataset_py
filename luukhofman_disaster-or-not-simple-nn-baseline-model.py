# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import keras



import numpy as np 

import pandas as pd 

import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
# Data import

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



tweets = [tweet for tweet in train_data['text']]

targets = [target for target in train_data['target']]

test_tweets = [tweet for tweet in test_data['text']]



# Split between test & validation -> 10% used for validation

sentences_train, sentences_val, y_train, y_val = train_test_split(tweets, targets, test_size=0.1, random_state=42)
# Initiate a vector of words 

# -> Max_df: I used this as a way to filter out recurring features in tweets (selects features that occur in max 80% of tweets)

# -> Min_df: I used this a way filter out terms at occur in less than 0.1% of the tweets



vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3), max_df=0.8, min_df=0.001)

vectorizer.fit(sentences_train)



x_train = vectorizer.transform(sentences_train)

x_val = vectorizer.transform(sentences_val)
# Specify input dimension

input_dimension = x_train.shape[1]



# Initiate a simple NN model 

model = keras.models.Sequential()

model.add(keras.layers.Dense(1500, input_dim=input_dimension, activation='relu'))

model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(500, input_dim=input_dimension, activation='relu'))

model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(100, input_dim=input_dimension, activation='relu'))

model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(2, input_dim=input_dimension, activation='softmax'))



# Get model summary

model.summary()



# Show if GPU is avialable

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model.compile(optimizer = 'adam',

              loss ='sparse_categorical_crossentropy',

              metrics = ['accuracy'])



model.fit(x = x_train, y = y_train, 

          batch_size = 32, epochs = 3, 

          verbose = 1, validation_data = (x_val, y_val), 

          shuffle = True, workers = 6, 

          use_multiprocessing = True)
# Make predictions

x_test = vectorizer.transform(test_tweets)

predictions = model.predict_classes(x_test)



# Write submission

submission = {"id":None, "target":None}



submission["id"] = [id for id in test_data['id']]

submission["target"] = [prediction for prediction in predictions]



pd.DataFrame.from_dict(data=submission).to_csv('disaster_submission.csv', header=True, index=False)