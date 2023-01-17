bvegaus_seed = 7 # tribute to bvegaus
import numpy as np
np.random.seed(bvegaus_seed)
import tensorflow as tf
tf.random.set_seed(bvegaus_seed)
import random as python_random
python_random.seed(bvegaus_seed)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

import os

import sys; print(sys.version)
datadir = "../input/red-wine-quality-cortez-et-al-2009/"
datacsv = "winequality-red.csv"

data = pd.read_csv(datadir+datacsv)
data.head()
ul = 2 # "UpperQ"
ml = 1 # "MiddleQ"
ll = 0 # "LowerQ"

upper_quality_threshold = 7
lower_quality_threshold = 5

quality_column_name = "quality"
quality = data[quality_column_name]
label = [ul if q>=upper_quality_threshold else (ll if q<=lower_quality_threshold else ml) for q in quality]
data[quality_column_name] = label
data.head()
data[[quality_column_name]].apply(pd.value_counts)
data = data.sample(frac=1, random_state=bvegaus_seed).reset_index(drop=True) # dataset shuffle
data.head()
data.info()
# three-fold partition is an improvement on bvegaus
X = data.drop([quality_column_name], axis = 1)
y = data[quality_column_name]
X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=0.2, random_state=bvegaus_seed, shuffle=True, stratify=y)
X_validation, X_test, y_validation, y_test = train_test_split(X_vt, y_vt, test_size=0.5, random_state=bvegaus_seed, shuffle=True, stratify=y_vt)
dataset_size = len(data.index)
train_size = len(X_train.index)
validation_size = len(X_validation.index)
test_size = len(X_test.index)
print("Dataset size = %s (TRN) + %s (VAL) + %s (TST) = %s" % (train_size, validation_size, test_size, dataset_size))
# using transform when appropriate is an improvement on bvegaus
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)
def red(neuronas, X_a, X_b, y_a, y_b):
    model = Sequential()
    model.add(Dense(neuronas,input_dim=X_a.shape[1], activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_a, y_a, epochs=200, verbose=0)

    ## evaluate the model
    scores = model.evaluate(X_b, y_b)
    return (model.metrics_names[1], scores[1]*100)
for i in range(2, 21):
    print("\nNeuronas: %d" % i)
    print("%s: %6.2f%%" % red(i, X_train, X_validation, y_train, y_validation))
