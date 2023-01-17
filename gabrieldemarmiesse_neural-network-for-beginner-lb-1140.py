# Importing the importations 

import pandas as pd

import numpy as np

#import dataframe as df

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

from scipy import sparse

from keras.regularizers import l2, l1

from keras.models import load_model

import os

from keras.layers.advanced_activations import PReLU

from keras.optimizers import *

import keras

from keras.layers import *

from keras.models import Sequential



# To see the training error in real time during training.

#from keras import callbacks

#remote = callbacks.RemoteMonitor(root='http://localhost:9000')
## Reading the input data

raw_data = pd.read_csv("../input/train.csv",delimiter=",")

raw_data_test = pd.read_csv("../input/test.csv",delimiter=",")
raw_data.head()
raw_data_test["loss"] = pd.Series([0]*len(raw_data_test))

raw_data_test.head()
class Preprocess:

    

    def __init__(self, dataset_to_fit):

        self.encoder = []

        self.onehot = OneHotEncoder()

        self.maxloss = dataset_to_fit["loss"].max()

        for i, feature in enumerate(dataset_to_fit.columns):

            if 'cat' in feature:

                self.encoder.append(list(dataset_to_fit[feature].value_counts().index))

                

        temp_data = self.conversion1(dataset_to_fit)

        

        self.onehot.fit(temp_data[:,1:117])

    



    def conversion1(self,dataframe):

        new_dataframe = pd.DataFrame()

        

        for i, feature in enumerate(dataframe.columns):

            if 'cat' in feature:

                new_dataframe[feature]=dataframe[feature].map(lambda x: self.convert(i,x))

            else:

                new_dataframe[feature] = dataframe[feature]

        

        return new_dataframe.as_matrix()

    

    def conversion2(self,array):

        indexes = array[:,0]

        cat = self.onehot.transform(array[:,1:117])

        cont = array[:,117:-1]

        loss = array[:,-1]

                

        return indexes, sparse.hstack((cont,cat)).toarray(), loss

        

    def convert(self,i,element):

        _list = self.encoder[i-1]

        try:

            return _list.index(element)

        except ValueError:

            return 0

    

    

    def convert_back(self,i,element):

        return self.encoder[i-1][element]

        

        

    def process(self,dataframe):

        array = self.conversion1(dataframe)

        indexes, features,loss = self.conversion2(array)

        

        #loss = loss/self.maxloss

        

        # Each one of them is a numpy array

        return indexes, features, loss
prepro = Preprocess(raw_data)
indexes, features,loss = prepro.process(raw_data)
model = Sequential()

model.add(Dense(256,activation = "relu", W_regularizer = l2(.01), input_dim = 1153))

model.add(Dense(128,activation = "relu", W_regularizer = l2(.01)))

model.add(Dense(64,activation = "relu", W_regularizer = l2(.01)))

model.add(Dense(32,activation = "relu", W_regularizer = l2(.01)))

model.add(Dense(16,activation = "relu", W_regularizer = l2(.01)))

model.add(Dense(1, init = 'he_normal'))
#sgd = SGD(lr=0.02, momentum=0.0, decay=0.03)

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.003)



model.compile(optimizer=adam, loss='mae')
try:

    history = model.fit(features, loss, batch_size=64, nb_epoch=1, verbose=2, validation_split=0.1)

except KeyboardInterrupt:

    pass
model.save("relu_regularizer.m5")
model = load_model("relu_regularizer.m5")
indexes_test, features_test,loss_test = prepro.process(raw_data_test)
losses_predicted = model.predict(features_test)
indexes = list(indexes_test)

indexes = [int(x) for x in indexes]
loss_ = list(losses_predicted)

loss_ = [x[0] for x in loss_]
submission_list = zip(indexes, loss_)
submission_number = 0

while os.path.isfile("submission_" + str(submission_number) + ".csv"):

    submission_number += 1

    

    

f = open("submission_" + str(submission_number) + ".csv", 'w')

f.write("id,loss\n")

for tup in submission_list:

    f.write(str(tup[0]) + "," + str(tup[1]) + "\n")

f.close()

    

print("Submission file number " + str(submission_number) + " was created")