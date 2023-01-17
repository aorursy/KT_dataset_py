# Python standard libraries are loaded

from __future__ import print_function

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Keras libraries loaded

import keras

from keras import metrics

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam, RMSprop

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from keras.utils import plot_model

from keras.models import load_model
# The file is read in from a CSV file

kc_data_org = pd.read_csv("../input/kc_house_data.csv")
# Describes dataframe features

kc_data_org.info()
# Displays the first 10 rows of the kc_data_org dataframe.

kc_data_org.head(10)
# Checks the kc_data_org for any null values.

kc_data_org.isnull().sum().sum()
# Transform dates into year, month and day and select columns.

kc_data_org['sale_yr'] = pd.to_numeric(kc_data_org.date.str.slice(0, 4))

kc_data_org['sale_month'] = pd.to_numeric(kc_data_org.date.str.slice(4, 6))

kc_data_org['sale_day'] = pd.to_numeric(kc_data_org.date.str.slice(6, 8))



kc_data = pd.DataFrame(kc_data_org, columns=[

        'sale_yr','sale_month','sale_day',

        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',

        'condition','grade','sqft_above','sqft_basement','yr_built',

        'zipcode','lat','long','sqft_living15','sqft_lot15','price'])

label_col = 'price'

#Prints out statistical summary of features

print(kc_data.describe())
# Distribution plots of the numerical features in the kc_data dataframe are visualized.

distributions = kc_data.select_dtypes([np.int, np.float])

for i, col in enumerate(distributions.columns):

    plt.figure(i)

    sns.distplot(distributions[col])
# Scatter plot of house price vs zipcode.

plt.figure(figsize=(12,7))

sns.scatterplot(kc_data['price'],kc_data['zipcode'])

plt.title('Price VS Zipcode')

plt.xlabel('Price')

plt.ylabel('Zipcode')

plt.show()
# Scatter plot of house price vs year house was built.

plt.figure(figsize=(12,8))

sns.scatterplot(kc_data['price'],kc_data['yr_built'])

plt.title('Price VS Year Built')

plt.xlabel('Price')

plt.ylabel('Year Built')

plt.show()
# Scatter plot of year house was built vs the square feet of the lot.

plt.figure(figsize=(12,7))

sns.scatterplot(kc_data['yr_built'],kc_data['sqft_lot'])

plt.title('Year Built VS Square Feet of Lot')

plt.xlabel('Year Built')

plt.ylabel('Square Feet of Lot')

plt.show()
# Scatter plot of price vs number of bedrooms in a home.

plt.figure(figsize=(12,7))

sns.scatterplot(kc_data['price'],kc_data['bedrooms'])

plt.title('Price Vs Number of Bedrooms')

plt.xlabel('Price')

plt.ylabel('# of Bedrooms')

plt.show()
# Boxplot of house condition distribution is shown. The higher the number the better the condition.

sns.boxplot(kc_data['condition'])
# Boxplot of house prices

plt.figure(figsize=(12,5))

sns.boxplot(kc_data['price'])
# Function to split a range of data frame & array indeces into three sub-ranges.

def train_validate_test_split(df, train_part=.6, validate_part=.2, test_part=.2, seed=None):

    np.random.seed(seed)

    total_size = train_part + validate_part + test_part

    train_percent = train_part / total_size

    validate_percent = validate_part / total_size

    test_percent = test_part / total_size

    perm = np.random.permutation(df.index)

    m = len(df)

    train_end = int(train_percent * m)

    validate_end = int(validate_percent * m) + train_end

    train = perm[:train_end]

    validate = perm[train_end:validate_end]

    test = perm[validate_end:]

    return train, validate, test
# Split index ranges into three parts, however, ignore the third.

train_size, valid_size, test_size = (70, 30, 0)

kc_train, kc_valid, kc_test = train_validate_test_split(kc_data, 

                              train_part=train_size, 

                              validate_part=valid_size,

                              test_part=test_size,

                              seed=2017)
# Extract data for training and validation into x and y vectors.

kc_y_train = kc_data.loc[kc_train, [label_col]]

kc_x_train = kc_data.loc[kc_train, :].drop(label_col, axis=1)

kc_y_valid = kc_data.loc[kc_valid, [label_col]]

kc_x_valid = kc_data.loc[kc_valid, :].drop(label_col, axis=1)



print('Size of training set: ', len(kc_x_train))

print('Size of validation set: ', len(kc_x_valid))

print('Size of test set: ', len(kc_test), '(not converted)')
# Function to get statistics about a data frame.

def norm_stats(df1, df2):

    dfs = df1.append(df2)

    minimum = np.min(dfs)

    maximum = np.max(dfs)

    mu = np.mean(dfs)

    sigma = np.std(dfs)

    return (minimum, maximum, mu, sigma)
# Function to Z-normalise the entire data frame - note stats for Z transform passed in.

def z_score(col, stats):

    m, M, mu, s = stats

    df = pd.DataFrame()

    for c in col.columns:

        df[c] = (col[c]-mu[c])/s[c]

    return df
# Normalize training and validation predictors using the stats from training data only

# (to ensure the same transformation applies to both training and validation data),

# and then convert them into numpy arrays to be used by Keras.

stats = norm_stats(kc_x_train, kc_x_valid)

arr_x_train = np.array(z_score(kc_x_train, stats))

arr_y_train = np.array(kc_y_train)

arr_x_valid = np.array(z_score(kc_x_valid, stats))

arr_y_valid = np.array(kc_y_valid)



print('Training shape:', arr_x_train.shape)

print('Training samples: ', arr_x_train.shape[0])

print('Validation samples: ', arr_x_valid.shape[0])

# Three functions to define alternative Keras models

#The first is very simple, consisting of three layers and Adam optimizer



def basic_model_1(x_size, y_size):

    t_model = Sequential()

    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))

    t_model.add(Dense(50, activation="relu"))

    t_model.add(Dense(y_size))

    print(t_model.summary())

    t_model.compile(loss='mean_squared_error',

        optimizer=Adam(),

        metrics=[metrics.mae])

    return(t_model)
# The second with Adam optimizer consists of 4 layers and the first uses 10% dropouts.

def basic_model_2(x_size, y_size):

    t_model = Sequential()

    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))

    t_model.add(Dropout(0.1))

    t_model.add(Dense(50, activation="relu"))

    t_model.add(Dense(20, activation="relu"))

    t_model.add(Dense(y_size))

    print(t_model.summary())

    t_model.compile(loss='mean_squared_error',

        optimizer=Adam(),

        metrics=[metrics.mae])

    return(t_model)
# The third is the most complex, it extends the previous model with Nadam optimizer, dropouts and L1/L2 regularisers.

def basic_model_3(x_size, y_size):

    t_model = Sequential()

    t_model.add(Dense(80, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))

    t_model.add(Dropout(0.2))

    t_model.add(Dense(120, activation="relu", kernel_initializer='normal', 

        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))

    t_model.add(Dropout(0.1))

    t_model.add(Dense(20, activation="relu", kernel_initializer='normal', 

        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))

    t_model.add(Dropout(0.1))

    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))

    t_model.add(Dropout(0.0))

    t_model.add(Dense(y_size))

    t_model.compile(

        loss='mean_squared_error',

        optimizer='nadam',

        metrics=[metrics.mae])

    return(t_model)
# Now we create the model - The code below will run basic_model_3

model = basic_model_3(arr_x_train.shape[1], arr_y_train.shape[1])

model.summary()
# Define how many epochs of training should be done and what is the batch size

epochs = 500

batch_size = 128



print('Epochs: ', epochs)

print('Batch size: ', batch_size)
# Specify Keras callbacks which allow additional functionality while the model is being fitted

# ModelCheckpoint allows to save the models as they are being built or improved

# TensorBoard interacts with TensorFlow interactive reporting system

# EarlyStopping watches one of the model measurements and stops fitting when no improvement



keras_callbacks = [

    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2)

    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)

    # TensorBoard(log_dir='/tmp/keras_logs/model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),

    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)

]
# Fit the model and record the history of training and validation.

# As we specified EarlyStopping with patience=20, with luck the training will stop in less than 200 epochs.

# Be patient, the fitting process takes time, use verbose=2 for visual feedback

history = model.fit(arr_x_train, arr_y_train,

    batch_size=batch_size,

    epochs=epochs,

    shuffle=True,

    verbose=0, # Change it to 2, if wished to observe execution

    validation_data=(arr_x_valid, arr_y_valid),

    callbacks=keras_callbacks)
train_score = model.evaluate(arr_x_train, arr_y_train, verbose=0)

valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=0)



print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4)) 

print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))
# Using matplotlib to create a plot of performance history 

def plot_hist(h, xsize=6, ysize=10):

    # Prepare plotting

    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = [xsize, ysize]

    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    

    # summarize history for MAE

    plt.subplot(211)

    plt.plot(h['mean_absolute_error'])

    plt.plot(h['val_mean_absolute_error'])

    plt.title('Training vs Validation MAE')

    plt.ylabel('MAE')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper left')

    

    # summarize history for loss

    plt.subplot(212)

    plt.plot(h['loss'])

    plt.plot(h['val_loss'])

    plt.title('Training vs Validation Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper left')

    

    # Plot it all in IPython (non-interactive)

    plt.draw()

    plt.show()



    return
plot_hist(history.history, xsize=8, ysize=12)