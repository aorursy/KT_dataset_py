from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import tensorflow_hub as hub

# Make numpy values easier to read.
np.set_printoptions(precision=2, suppress=True)
a = "hello"
#import and shuffle
df = pd.read_csv("../input/turbotax-reviews-2018/turbotax_reviews_2018.csv")
df = df.sample(frac=1)
#import embedding hub layer
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

#create embedding wrapper function
def embed(x):
    return pd.Series(hub_layer(np.array([x], dtype=str)).numpy()[0])

#embed text layer
embed_length = 20
col_start = 4
new_cols = []

#generate the column list:
for x in range(embed_length):
    
    new_cols.append(col_start + x)


#add the embeded columns
df[new_cols] = df['review'].apply(embed)

#drop the review column
df = df.drop(columns=['review'])

df.head()
#split into test and training data
train_data_split = .6
split = round(train_data_split * len(df))
train_labels = np.array(df.iloc[:split,2])
test_labels = np.array(df.iloc[split+1:,2])

df = df.drop(columns=['label'])

train_data = np.array(df.iloc[:split])
test_data = np.array(df.iloc[split+1:])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, input_shape=[22], activation=tf.nn.relu),
    tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)

])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['mae', 'accuracy'],

)
epochs = 100
history = model.fit(train_data, train_labels, epochs=epochs, callbacks=[early_stop], verbose=True)
model.evaluate(test_data,  test_labels, verbose=2)
predictions = model.predict(test_data)
categories = ['Happy','Pricing','Functionality','Product','Tech','Misc','Support']

def classify(platform,review,rating):
    
    vectors = []
    
    if(platform == "iOS"):
        platform=1     
    else:
        platform=0
    
    vectors.append(platform)
    vectors.append(rating/5)
    vectors = vectors + embed(review).values.tolist()
    
    prediction = np.array([vectors])
    
    return categories[model.predict(prediction).argmax()]
    

classify("iOS", "You suck, TurboTax", 1)

weekly = pd.read_csv("../input/weekly-app-data/weekly.csv")
weekly = weekly.drop(columns="Unnamed: 4")
weekly.head()
weekly['Category'] = weekly.apply(lambda row: classify(row['Reviews'], row['Reviews Content'], row['Rating']) , axis=1)
weekly.head()