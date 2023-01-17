import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 10,8

sns.set(style='whitegrid', palette='muted',

        rc={'figure.figsize': (15,10)})

import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

import tensorflow as tf

def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)

# Program Main Execution Starts From Here

# Print Datasets found

print(os.listdir("../input/homedataset"), '\n')



# Get Two CSV's and read their data using pandas object and load them the training to "train" and testing to "test"

train = pd.read_csv('../input/homedataset/trainHome.csv', )

test = pd.read_csv('../input/homedataset/testHome.csv')



# Print Read Data from train (To make sure it read the file correctly)

print('Printing Train Data: \n', train, '\n')

print('Printing Test Data: \n',test, '\n')

print('Is there null values in training data?\n', train.isnull().sum(), '\n') # To check there are no null values

print('Is there null values in testing data?\n', test.isnull().sum(), '\n') # To check there are no null values

# Display all training data with statistical data

display_all(train.describe(include='all').T)



# Display all training data with statistical data

display_all(test.describe(include='all').T)
# Plot the relation between rate and price

#sns.countplot(x='rate(1-10)', data=train, palette='hls', hue='price (jd)')

plt.scatter(train['rate(1-10)'], train['price (jd)'])

plt.show()



# Plot the relation between area and price

#sns.countplot(x='area m2', data=train, palette='hls', hue='price (jd)')

plt.scatter(train['area m2'], train['price (jd)'])

plt.show()



# Plot a graph to see the correlation between price, area, and rank

correlation = train.corr()

sns.heatmap(correlation)
# Message continous and discrete data both are treated as continous

continousDiscreteData = ['area m2', 'rate(1-10)']

scaler = MinMaxScaler() # You can also use StandardScaler for improved results

for data in continousDiscreteData:

    # Transform Train Data For Model Trainging

    train[data] = train[data].astype('float64')

    train[data] = scaler.fit_transform(train[[data]])

    # Transform Test Data For Validation

    test[data] = test[data].astype('float64')

    test[data] = scaler.fit_transform(test[[data]]) 

    

print('Print train data after massaging: \n', train, '\n')

print('Print test data after massaging: \n', test, '\n')



display_all(train.describe(include='all').T)

display_all(test.describe(include='all').T)
# Train Data

X_train = train[pd.notnull(train['price (jd)'])].drop(['price (jd)'], axis=1)

y_train = train[pd.notnull(train['price (jd)'])]['price (jd)']



# Validation Data

X_test = test[pd.notnull(test['price (jd)'])].drop(['price (jd)'], axis=1)

y_test = test[pd.notnull(test['price (jd)'])]['price (jd)']



print('X_train: \n', X_train, '\n')

print('y_train: \n', y_train, '\n')

print('X_test: \n', X_test, '\n')

print('y_test: \n', y_test, '\n')
# Model Creation with 8 layers where activation functions are sigmoid and optimization is adaptive with momentum

def create_model(lyrs=8, act='relu', opt='Adam', dr=0.0):

    

    model = Sequential()

    

    # create first hidden layer

    # 'normal' distribution used to initialize weights.

    model.add(Dense(256, kernel_initializer='normal', input_dim=X_train.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(0, lyrs):

        # Each layer has 256 neurons

        model.add(Dense(256, activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, kernel_initializer='normal'))  # output layer

    

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

    

    return model
# Create Model

model = create_model()

print(model.summary())

# Training Model

training = model.fit(X_train, y_train, epochs=500, batch_size=3, validation_split=0.6, verbose=0, validation_data=(X_test, y_test))

#training = model.fit(X_train, y_train, epochs=500, batch_size=2, validation_split=0.25, verbose=0)

#print(training.history.keys()) #To know what are the dictionary parameters returned

mae = np.mean(training.history['val_mean_absolute_error'])

print("\n%s: %.2f%%" % ('Mean Absolute Error', mae))
# Show model accuracy on graph for comparison

plt.plot(training.history['mean_absolute_error'])

plt.plot(training.history['val_mean_absolute_error'])

plt.title('Model Train/Validation Comparison')

plt.ylabel('Accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()



# Show model loss on graph for comparison

plt.plot(training.history['loss'])

plt.plot(training.history['val_loss'])

plt.title('Model Train/Validation Comparison')

plt.ylabel('Loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# Predict On Test Data

test['price (jd)'] = model.predict(X_test)

test['price (jd)'] = test['price (jd)'].apply(lambda x: round(x,0)).astype('float64')



# Prepare The Wanted Data And Display It

solution = test[['price (jd)']]

solution.head()