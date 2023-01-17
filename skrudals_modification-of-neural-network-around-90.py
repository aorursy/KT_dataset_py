# based on https://www.kaggle.com/vbmokin/heart-disease-comparison-of-20-models

# modified by Eric Na (Skrudals), 2, March, 2020



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import pandas_profiling as pp



from sklearn import metrics



# NN models

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import optimizers

from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping, ModelCheckpoint



# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
data = pd.read_csv("../input/heart-disease-uci/heart.csv") # kaggle



target_name = 'target'

data_target = data[target_name]

data = data.drop([target_name], axis=1)



train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)
# get mean and std from training data

mean = np.mean(train)

std = np.std(train)



# normalization

train = (train-mean)/(std+1e-7)

test = (test-mean)/(std+1e-7)



# split training set to validation set

Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.2, random_state=0)

# As NN is sensitive to its initialization and train_test_split splits validation set randomly,

# the results may vary for each trial...

# and so suggest to run several times...

# Typically, about 92-93% for training data and about 90% (sometimes, over 91%, but sometimes 86~87%) for testing data

def build_ann(optimizer='adam'):



    # Initializing the ANN

    ann = Sequential()



    # Adding the input layer and the first hidden layer of the ANN

    ann.add(Dense(units=32, kernel_initializer='he_normal', activation='relu', input_shape=(len(train.columns),)))

    # Adding the output layer

    ann.add(Dense(units=1, kernel_initializer='he_normal', activation='sigmoid'))



    # Compiling the ANN

    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



    return ann



opt = optimizers.Adam(lr=0.001)

ann = build_ann(opt)



# Training the ANN

history = ann.fit(Xtrain, Ztrain, batch_size=16, epochs=200, validation_data=(Xval, Zval))



# Predicting the Train set results

ann_prediction = ann.predict(train)

ann_prediction = (ann_prediction > 0.5)*1 # convert probabilities to binary output



# Compute error between predicted data and true response and display it in confusion matrix

acc_ann1 = round(metrics.accuracy_score(target, ann_prediction) * 100, 2)

print(acc_ann1)



# Predicting the Test set results

ann_prediction_test = ann.predict(test)

ann_prediction_test = (ann_prediction_test > 0.5)*1 # convert probabilities to binary output



# Compute error between predicted data and true response and display it in confusion matrix

acc_test_ann1 = round(metrics.accuracy_score(target_test, ann_prediction_test) * 100, 2)

print(acc_test_ann1)