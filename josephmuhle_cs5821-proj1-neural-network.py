# loading the requred packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense # for fully connected layers

#load in training data
df_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv") # read csv and convert to dataframe
exp = df_train.loc[: ,"label"] # copy labels from data
data = df_train.loc[:, df_train.columns != "label"] # seperate labels from data
dummies = pd.get_dummies(exp) # changes label represention (ex. 2 as 0010000000)

#load in testing data
df_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv") # read csv and convert to dataframe
exp_test = df_test.loc[: ,"label"] # copy labels from data
data_test = df_test.loc[:, df_test.columns != "label"] # seperate labels from data
dummies_test = pd.get_dummies(exp_test) # changes label represention (ex. 2 as 0010000000)
#create model
model = Sequential()
model.add(Dense(10, activation = "relu", input_dim = 784)) # first layer
model.add(Dense(9, activation = "relu")) #second layer
model.add(Dense(8, activation = "relu")) #third layer
model.add(Dense(10, activation = "softmax")) #output layer (origionally used sigmoid, but that caused errors in later epochs)
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy", 
    metrics = ["accuracy"]
) # compile model
# train the model
model.fit(data, dummies, 
          epochs = 40, 
          verbose = 0 
         ) 
#checking model performance
scores = model.evaluate(data, dummies)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(data_test, dummies_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))
#load in training data
df_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv") # read csv and convert to dataframe
exp = df_train.loc[: ,"label"] # copy labels from data
data = df_train.loc[:, df_train.columns != "label"] # seperate labels from data
dummies = pd.get_dummies(exp) # changes label represention (ex. 2 as 0010000000)

#load in testing data
df_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv") # read csv and convert to dataframe
exp_test = df_test.loc[: ,"label"] # copy labels from data
data_test = df_test.loc[:, df_test.columns != "label"] # seperate labels from data
dummies_test = pd.get_dummies(exp_test) # changes label represention (ex. 2 as 0010000000)
#create model
fashion_model = Sequential()
fashion_model.add(Dense(10, activation = "relu", input_dim = 784)) # first layer
fashion_model.add(Dense(9, activation = "relu")) #second layer
fashion_model.add(Dense(8, activation = "relu")) #third layer
fashion_model.add(Dense(10, activation = "softmax")) #output layer (origionally used sigmoid, but that caused errors in later epochs)
fashion_model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy", 
    metrics = ["accuracy"]
) # compile model


# training the model
fashion_model.fit(data, dummies, 
          epochs = 60,
          verbose = 0
         ) 
#checking model performance
scores = fashion_model.evaluate(data, dummies)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = fashion_model.evaluate(data_test, dummies_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))