# This Python 3 environment comes with many helpful analytics libraries install

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Importing all of Keras Modules

import keras

from keras import models

from keras import layers

from keras.layers import LeakyReLU

from keras.layers import ELU

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
# Reading the train and test data



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Creating a copy of the training data set to conduct EDA:



train_copy = train.copy()

test_copy = test.copy()
# The shape of the data sets are:



print("training data contains ", train_copy.shape[0]," records and ",train_copy.shape[1], " columns \n")

print("training data contains ", test.shape[0]," records and ",test.shape[1], " columns \n")
# Taking a look at the training data



train_copy.info()
# Collecting summary statistics for all the columns:



train_copy.describe()
#columns = list(train_copy)

#for i in columns:

#    if train_copy[i].dtype != np.int64 and train_copy[i].dtype != np.float64:

#        print(train_copy[i].value_counts())

        
# Dropping the sparse columns, ID (from both Trainig and test)

train_copy.drop(["Id","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis = 1, inplace=True)

test_copy.drop(["Id","Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis = 1, inplace=True)
train_copy.head()
test_copy.head()
# Checking for Ordinal Variables

columns = list(train_copy)

for i in columns:

    if train_copy[i].dtype != np.int64 and train_copy[i].dtype != np.float64:

        print(train_copy.groupby(i)['SalePrice'].agg({"average_price":"mean", "#Entries":"count"}).sort_values(by=['average_price']))
# Missing value treatment. Replacing String Columns with 'NAs' and Numerical Columns with '-99' (junk value)



#Training data

columns_train = list(train_copy)

for i in columns_train:

    if train_copy[i].dtype == 'object':

        train_copy[i] = train_copy[i].fillna('Na')

    else:

        train_copy[i] = train_copy[i].fillna(-99)
# Missing value treatment. Replacing String Columns with 'NAs' and Numerical Columns with '-99' (junk value)



# Test data

columns_test = list(test_copy)

for i in columns_test:

    if test_copy[i].dtype == 'object':

        test_copy[i] = test_copy[i].fillna('Na')

    else:

        test_copy[i] = test_copy[i].fillna(-99)
# Seperating examples and labels from the training data set

X =  train_copy.drop(['SalePrice'], axis=1)

y = np.log(train_copy.SalePrice)
X.head()
# Doing one hot encoding for both training and test data sets



X = pd.get_dummies(X)

test_copy = pd.get_dummies(test_copy)
print("one hot encoded training data set shape", X.shape, "\n")

print("one hot encoded training data set shape", test_copy.shape, "\n")
common_cols = [a for a in list(train_copy) for b in list(test_copy) if a==b]

len(common_cols)
# Subsetting both train and test data sets for the common columns:

train_copy_comm = X[common_cols]

test_copy_comm = test_copy[common_cols]
# Standardizing the training and test data set

mean = train_copy_comm.mean(axis = 0)

std = train_copy_comm.std(axis = 0)



train_copy_comm -= mean

train_copy_comm /= std



test_copy_comm -= mean

test_copy_comm /= std
# The above operation has introduced NAs in the data set becuase of the division operation. We need to do the null value imputation again

train_copy_comm = train_copy_comm.fillna(0)

test_copy_comm = test_copy_comm.fillna(0)
train_copy_comm.head()
train_data, validation_data, train_targets, validation_targets = train_test_split(train_copy_comm, y,test_size=0.2)
# Building the network



# Model Build

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred-y_true)))

    

def build_model_dropout():

    model = models.Sequential()

    model.add(layers.Dense(240,activation='relu', 

                           input_shape=(train_copy_comm.shape[1],)))

    model.add(layers.Dropout(0.45))

    model.add(layers.Dense(240,activation='relu'))

    model.add(layers.Dropout(0.45))

    model.add(layers.Dense(240,activation='relu'))

    model.add(layers.Dropout(0.45))

    model.add(layers.Dense(120,activation='relu'))

    model.add(layers.Dropout(0.35))

    model.add(layers.Dense(120,activation='relu'))

    model.add(layers.Dropout(0.35))

    model.add(layers.Dense(60,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(60,activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss=root_mean_squared_error, metrics=['mse'])

    model.summary()

    return model
# Trying out iterated K-Fold Validation with Shuffling



num_epochs = 900

all_mse_histories = []





# Invoking the model function

model = build_model_dropout()



#Creating a checkpoint for the network

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



#Defining a callback function to reduce the learning rate near the minim

reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.9,patience=50, min_lr=0.001,  verbose=1)



#Defining a callback function for early stopping

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)



#All the callbacks list

callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]



history = model.fit(train_data, train_targets, validation_data=(validation_data,validation_targets), epochs=num_epochs, 

                callbacks=callbacks_list, batch_size=20,verbose=2)



mse_history = history.history['val_mean_squared_error']

all_mse_histories.append(mse_history)

average_mse_history = history.history['val_mean_squared_error']
def smooth_curve(points, factor=0.9):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous*factor + point*(1-factor))

        else:

            smoothed_points.append(point)

    return smoothed_points
smooth_mse_history = smooth_curve(average_mse_history)



plt.plot(range(1, len(smooth_mse_history) + 1), smooth_mse_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
# load the model

from keras.models import Sequential, load_model

new_model = load_model("weights.best.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})
# estimate accuracy on whole dataset using loaded weights

scores = new_model.evaluate(validation_data, validation_targets, verbose=0)

print("%s: %.4f%%" % (new_model.metrics_names[1], scores[1]))

print("%s: %.4f%%" % (new_model.metrics_names[0], scores[0]))

#0.1330
predictions = new_model.predict(test_copy_comm)

submission = pd.read_csv('../input/sample_submission.csv')

submission.SalePrice = np.exp(predictions)

submission.to_csv('submission.csv', index=False)