DATA_DIR = "../input/applications-of-deep-learningwustl-spring-2020/"



import numpy as np

import pandas as pd

import os

from tensorflow.keras.utils import to_categorical



print("Loading original data...")

df = pd.read_csv(DATA_DIR+"train.csv")

X = np.array(df.drop(['id','glasses'], axis=1))

y = np.array(to_categorical(df['glasses']))

print("Done!")
print(X.shape, y.shape)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization



print("Creating model...")



model = Sequential([

    Dense(512, input_shape=X.shape[1:]), BatchNormalization(), LeakyReLU(0.1), Dropout(0.1),

    Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3),

    Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3),

    Dense(512), BatchNormalization(), LeakyReLU(0.1), Dropout(0.3), 

    Dense(256), BatchNormalization(), LeakyReLU(0.1), 

    Dense(y.shape[:][1], activation='softmax')    

])



model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

print("Model created.")
from tensorflow.keras.callbacks import ModelCheckpoint



mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)



model.fit(X, y, 

          epochs=1000, 

          validation_split=0.2, 

          callbacks=[mc])



print("Model trained.")
print("Loading best model...")

model.load_weights("best_model.h5")



print("Loading test data...")

df = pd.read_csv(DATA_DIR+"test.csv")

X = np.array(df.drop(['id'], axis=1))

y = model.predict_proba(X)

df['glasses'] = y[:,1]

df['glasses'] = df['glasses']

df[['id','glasses']].to_csv("submission.csv", index=False)



print(df[['id','glasses']].head(20))

print("Done!")