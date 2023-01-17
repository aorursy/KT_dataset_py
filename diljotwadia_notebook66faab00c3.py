import pandas as pd
dataset = pd.read_csv(r'../input/cat-in-the-dat/train.csv')

testset = pd.read_csv(r'../input/cat-in-the-dat/test.csv')
X = dataset.iloc[:,1:24].values

Y=dataset.iloc[:,24].values

testX = testset.iloc[:,1:].values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
for i in range(3,15):

  labelEncoder=LabelEncoder()

  X[:,i]=labelEncoder.fit_transform(X[:,i]) 

for i in range(16,21):

  labelEncoder=LabelEncoder()

  X[:,i]=labelEncoder.fit_transform(X[:,i]) 



for i in range(3,15):

  labelEncoder=LabelEncoder()

  testX[:,i]=labelEncoder.fit_transform(testX[:,i]) 

for i in range(16,21):

  labelEncoder=LabelEncoder()

  testX[:,i]=labelEncoder.fit_transform(testX[:,i]) 
import tensorflow as tf

from tensorflow import keras
model = keras.models.Sequential([

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid'),

])
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
import numpy as np
model.fit(np.asarray(X).astype(np.int32), np.asarray(Y).astype(np.int32), epochs=30, batch_size=100, verbose=1)

predictions = model.predict(np.asarray(testX).astype(np.int32))
submit = pd.concat([testset['id'], pd.Series(predictions[:,0]).rename('target')], axis=1)

submit.to_csv('submission.csv', index=False, header=True)
submit