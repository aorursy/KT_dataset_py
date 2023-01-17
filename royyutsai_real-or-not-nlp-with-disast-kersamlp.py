# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])

print(train_vectors.shape)

print(train_vectors[0].todense().shape)

print(train_vectors[0].todense())

print(test_vectors[0].todense().shape)

## Train

# input: train_vectors

# output: train_df["target"]

## Test

# input: test_vectors

# output: sample_submission["target"]
train_x = train_vectors[:-800]

train_y = train_df["target"][:-800]

test_x = train_vectors[7000:7600]

test_y = train_df["target"][7000:7600]
# clf.fit(train_vectors, train_df["target"])

# sample_submission["target"] = clf.predict(test_vectors)

# print(sample_submission.head())

# sample_submission.to_csv("submission.csv", index=False)
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(50, input_dim=train_vectors[0].todense().shape[1], activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



## Train

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, batch_size=5)

_, accuracy = model.evaluate(test_x, test_y)

print('Accuracy: %.2f' % (accuracy*100))
predictions = model.predict_classes(test_vectors)

# for i in range(5):

# 	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

sample_submission["target"] = predictions

print(sample_submission.head())

print(sample_submission[0:20])

sample_submission.to_csv("submission.csv", index=False)