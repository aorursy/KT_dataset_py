import numpy as np

import pandas as pd

import keras

from keras.models import Model

from keras.layers import *

from keras import optimizers

import tensorflow as tf
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)

#42000 rows and 785 columns

print(df_test.shape)

#28000 rows in test
df_train.head()

#784 features and 1 label 
#iloc The iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position

#iloc basically used for dataframe slicing

df_features = df_train.iloc[:, 1:785]

print(df_features.shape)



#extracting al lables tha contains our actual data

df_label = df_train.iloc[:, 0]

print(df_label.shape)



X_test = df_test.iloc[:, 0:784]



print(X_test.shape)
#splitting data into train and test

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(df_features,df_label,

                                          test_size=0.2,

                                          random_state=1212)



X_train = X_train.as_matrix().reshape(33600, 784) #(33600, 784)

X_val = X_val.as_matrix().reshape(8400, 784) #(8400, 784)



X_test = X_test.as_matrix().reshape(28000, 784)

X_train=tf.keras.utils.normalize(X_train,axis=1)

X_test=tf.keras.utils.normalize(X_test,axis=1)

print((min(X_train[1]), max(X_train[1])))

#normalizing the values same as above code

#convert labels to one hot encoders

num_digits = 10

#Converts a class vector (integers) to binary class matrix.

y_train = keras.utils.to_categorical(y_train, num_digits)

y_val = keras.utils.to_categorical(y_val, num_digits)



print(y_train[2])
#Model fitting 

#building a neural network of 5 layers

model=tf.keras.models.Sequential()

#creating input layer

#model.add(tf.keras.layers.Flatten())



#hidden layers creation

model.add(tf.keras.layers.Dense(300,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(200,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(200,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(200,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(100,activation=tf.nn.relu))



#output layer

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

training_epochs = 30

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

training_model = model.fit(X_train, y_train,

                     epochs = training_epochs,

                     verbose = 2,

                     validation_data=(X_val, y_val))
predictions=model.predict(X_test)

print(predictions)
print(np.argmax(predictions[11]))
#pd.dataframe converts output to a dataframe

pred = pd.DataFrame(model.predict(X_test))

pred = pd.DataFrame(pred.idxmax(axis = 1))

pred.index.name = 'ImageId'

pred = pred.rename(columns = {0: 'Label'}).reset_index()

pred['ImageId'] = pred['ImageId'] + 1



pred.head(20)
pred.to_csv('final_submission.csv', index = False)