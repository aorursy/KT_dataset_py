import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Admission_Predict.csv")



# Print the head of df

print(df.head())



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
#Scaling the continuos variables

df_scale = df.copy()

scaler = preprocessing.StandardScaler()

columns =df.columns[1:7]

df_scale[columns] = scaler.fit_transform(df_scale[columns])

df_scale.head()

df_scale = df_scale.iloc[:,1:9]
sample = np.random.choice(df_scale.index, size=int(len(df_scale)*0.8), replace=False)

train_data, test_data = df_scale.iloc[sample], df_scale.drop(sample)



print("Number of training samples is", len(train_data))

print("Number of testing samples is", len(test_data))

print(train_data[:10])

print(test_data[:10])
features = train_data.drop('Chance of Admit ', axis=1)

targets = train_data['Chance of Admit ']

targets = targets > 0.5

features_test = test_data.drop('Chance of Admit ', axis=1)

targets_test = test_data['Chance of Admit ']

targets_test = targets_test > 0.5
# Activation (sigmoid) function

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):

    return sigmoid(x) * (1-sigmoid(x))

def error_formula(y, output):

    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(y, output):

    return (y-output) * output * (1 - output)
# Neural Network hyperparameters

epochs = 9000

learnrate = 0.3



# Training function

def train_nn(features, targets, epochs, learnrate):

    

    # Use to same seed to make debugging easier

    np.random.seed(42)



    n_records, n_features = features.shape

    last_loss = None



    # Initialize weights

    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    print(weights.shape)



    for e in range(epochs):

        del_w = np.zeros(weights.shape)

        for x, y in zip(features.values, targets):

            # Loop through all records, x is the input, y is the target



            # Activation of the output unit

            #   Notice we multiply the inputs and the weights here 

            #   rather than storing h as a separate variable 

            output = sigmoid(np.dot(x, weights))



            # The error, the target minus the network output

            error = error_formula(y, output)



            # The error term

            #   Notice we calulate f'(h) here instead of defining a separate

            #   sigmoid_prime function. This just makes it faster because we

            #   can re-use the result of the sigmoid function stored in

            #   the output variable

            error_term = error_term_formula(y, output)



            # The gradient descent step, the error times the gradient times the inputs

            del_w += error_term * x



        # Update the weights here. The learning rate times the 

        # change in weights, divided by the number of records to average

        weights += learnrate * del_w / n_records



        # Printing out the mean square error on the training set

        if e % (epochs / 10) == 0:

            out = sigmoid(np.dot(features, weights))

            loss = np.mean((out - targets) ** 2)

            print("Epoch:", e)

            if last_loss and last_loss < loss:

                print("Train loss: ", loss, "  WARNING - Loss Increasing")

            else:

                print("Train loss: ", loss)

            last_loss = loss

            print("=========")

    print("Finished training!")

    return weights

    

weights = train_nn(features, targets, epochs, learnrate)
# Calculate accuracy on test data

tes_out = sigmoid(np.dot(features_test, weights))

predictions = tes_out > 0.5

#predictions = tes_out

accuracy = np.mean((predictions == targets_test))

print("Prediction accuracy: {:.3f}".format(accuracy))
features = train_data.drop('Chance of Admit ', axis=1)

targets = train_data['Chance of Admit ']

features_test = test_data.drop('Chance of Admit ', axis=1)

targets_test = test_data['Chance of Admit ']
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.utils import np_utils



# Building the model

model = Sequential()

model.add(Dense(1, activation='softmax', input_shape=(7,)))

model.add(Dense(1, activation='softmax'))



# Compiling the model

model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()
# Training the model

model.fit(features, targets, epochs=9000, batch_size=1, verbose=0)