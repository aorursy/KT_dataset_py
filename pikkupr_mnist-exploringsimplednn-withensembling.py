# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np 

import pandas as pd



# to get the accuracy score, provided actual class labels and predicted labels

from sklearn.metrics import accuracy_score



# to split the dataset into training and validation set

from sklearn.model_selection import train_test_split



# for reproducibility purposes

seed = 42

np.random.RandomState(seed)



print(os.listdir("../input"))
# importing the various Kers modules



# We will be using the SGD optimization technique.

# This method calculate the cost function for each datapoint individually.

# Unlike th normal GD, in which error is calculated on the entire dataset altogether.

from keras.optimizers import SGD 



# This is used to create a linear stack of layers, starting from input layer, 

# followed by hidden layers  and ending with the output layer.

from keras.models import Sequential



# to convert the categorical data into One-hot encoding vector

from keras.utils import to_categorical



# Dense is to create a single layer in a network, consist of input shape, number of nodes in the layer

# and the activation function to use for the nodes in the layer



# Dropout is to specify the dropout rate after each layer

from keras.layers import Dense, Dropout



# History is used to get the various information about the training of the model, 

# like the training and test accuracies at the each epoch



# EarlyStopping help us to stop the training is the model is not showing significant improvement,

# in certain number of epochs

from keras.callbacks import EarlyStopping, History
# to read the training data i.e. the MNIST image data

train_df = pd.read_csv("../input/train.csv")

train_df.head(5)
print("Shape of the training dataset: ", train_df.shape)
# seperating the labels from the image pixels information

y = train_df.values[:, 0] # get the values in the first column

X = train_df.values[:, 1:]/255.0 # get all the columns but the first. Also, we divided by 255 to normalize the values
# create dummy variables from the labels using One-hot encoding method, which we imported above

y_encoded = to_categorical(y)

print("Shape of the target variable: ", y_encoded.shape)



print("First few records:")

print(y_encoded[:2])
# split the data into train and validation datasets with test size to be 30% of the total

validation_split = .3



# stratify here ensures that the proportion of the labels is maintained in the train & test dataset

# random_state is specified so that the results can be replicated later on

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=validation_split, stratify=y, random_state=seed)
print("Size of the training dataset: ", X_train.shape[0])

print("Size of the testing dataset: ", X_val.shape[0])
def build_model(input_shape, num_classes, num_layers, num_nodes, activation="relu", optimizer="adam", dropout_rate=0.00):

    

    # this will contain all the layers in the network

    model = Sequential()

    

    # add a layer with "num_nodes" and input size of "input_size"

    model.add(Dense(num_nodes, activation=activation, input_shape=input_shape))

    

    # add the dropout layer

    model.add(Dropout(rate=dropout_rate))

    

    # to add the hidden-layers

    for i in range(num_layers-1):    

        model.add(Dense(num_nodes, activation=activation))

        model.add(Dropout(rate=dropout_rate))

    

    # add an output layer of size "num_classes" which is equal to the number of labels we've in the data

    # in case of MNIST it is 10

    # softmax function here converts the output into the probabilities i.e. in range [0,1]

    model.add(Dense(num_classes, activation="softmax"))

    

    # in our case we will be using SGD optimizer, loss/cost function will be categorical_crossentropy

    # the metrics we want our model to calculate at each epoch can also be passed

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    

    return model
# if we found that at each consecutive epoch model is not getting any better that is loss is not decreasing then there is no

# point in letting the model to keep training for the remainder of the epochs.

# in such cases EarlyStopping, helps us to stop the training if there is no incremental benefit

# Here, if the difference between the validation losses in two consecutive epochs is less than 1/1000

# the model will stop training

early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3)
# maximum number epochs/times to run the model

max_num_epochs = 50



# a list of learning rates 

learning_rate = [0.0001, 0.001]



# a list of % drop rates

dropouts = [0.05, 0.10]



# number of hidden layers to keep in the NN models

num_layers = [10, 12]



# number of nodes to keep in the hidden layer

num_nodes = [1000, 1200]
# to keep the accuracies of each of the model

model_accuracies = []



# iterate over the configurations and create a model with those configurations

for dropout in dropouts:

    for lr in learning_rate:

        for num_layer in num_layers:

            for num_node in num_nodes:

                

                # this keeps the history of the loss/val_loss for each of the epoch

                history = History()



                print("\n---- Model Configuration ----")

                print("Dropout rate: ", dropout)

                print("Learning rate: ", lr)

                print("Number of Layers: ", num_layer)

                print("Number of Nodes in hidden layers: ", num_node)

                

                # create and train the model for each of the configuration

                # here, SGD is being used

                model = build_model(input_shape=(784,), num_classes=10, num_layers=num_layer, num_nodes=num_node,  

                                    activation='relu', optimizer=SGD(lr=lr), dropout_rate=dropout)



                # both training and validation datasets are provided

                # we passing the history, early_stopping_monitor in the callbacks parameter

                # setting verbose as False, will stop the model from printing the training information for each epoch

                model.fit(X_train, y_train, epochs=max_num_epochs, callbacks=[history, early_stopping_monitor], 

                          validation_data=(X_val, y_val), verbose=False)

                

                print("Number of epochs: ", len(history.history['loss']))

                print("Train accuracy: ", model.evaluate(X_train, y_train, verbose=0)[1])

                print("Validation accuracy: ", model.evaluate(X_val, y_val, verbose=0)[1])



                # maintain the configurations and the accuracies to pick the best model at a later stage

                model_accuracies.append([

                    dropout,

                    lr,

                    num_layer,

                    num_node,

                    model.evaluate(X_train, y_train, verbose=0)[1],

                    model.evaluate(X_val, y_val, verbose=0)[1]

                ])
# picking up the best configuration, i.e. the model giving the best performance on the validation dataset

best_config = sorted(model_accuracies, key=lambda row: row[5], reverse=True)[0]
print("The model giving the best performance has the following configuration: ", best_config)

print("Number of layers in the network: ", best_config[2])

print("Number of nodes in the hidden layers: ", best_config[3])

print("Dropout rate: ", best_config[0])

print("Learning rate: ", best_config[1])
# Let's create 15 different models with same configuration

models = [ build_model(input_shape=(784,), num_classes=10, num_layers=best_config[2], num_nodes=best_config[3],  

           activation='relu', optimizer=SGD(lr=best_config[1]), dropout_rate=best_config[0]) for i in range(15)]
# train each of these models

# setting verbose as false - to prevent any information from getting printed

for i in range(15):

    models[i].fit(X_train, y_train, epochs=max_num_epochs, callbacks=[history, early_stopping_monitor], 

                  validation_data=(X_val, y_val), verbose=False)
# now, we will get the probabilities from each of the model

# sum them up and will get the class with the highest total probability 



val_probabs = np.zeros((y_val.shape[0], 10), dtype='float32') # create an array containing zeroes



for i in range(15):

    # predict and add the probabilities

    val_probabs += models[i].predict(X_val)

    

# get the actual labels of the validation set

acual_labels = y_val.argmax(axis=-1)



# get the predict labels

predicted_labels = val_probabs.argmax(axis=-1)



# print the validation accuracy

print("Validation accuracy from the Ensemble model: ", accuracy_score(acual_labels, predicted_labels))
# read the test dataset

test_df = pd.read_csv("../input/test.csv")

print("Shape of the test data: ", test_df.shape[0])

test_df.head(2)
# get the values as numpy array

X_test = test_df.values
# create an array containing zeroes

test_probabs = np.zeros((X_test.shape[0], 10), dtype='float32') 



for i in range(15):

    # predict and add the probabilities

    test_probabs += models[i].predict(X_test)

    

# get the actual labels of the validation set

predicted_classes = test_probabs.argmax(axis=-1)



print("Size of the predicted classes: ", predicted_classes.shape)
# create output dataframe and file

output = pd.DataFrame()



output["ImageId"] = [i for i in range(1, predicted_classes.shape[0]+1)]

output["Label"] = predicted_classes



output.to_csv("predicted_classes.csv", index=False)