#use packet

import numpy as np

import pandas as pd

import tensorflow as tf

from keras.models import Model, Sequential

from keras.layers import Input, Activation, Dense

from keras.optimizers import SGD

from keras.utils.np_utils import to_categorical

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Pandas read CSV

sf_train = pd.read_csv('/kaggle/input/dota-heroes/p5_training_data.csv')



# Correlation Matrix for target

corr_matrix = sf_train.corr()

print(corr_matrix['type'])



# Drop unnecessary columns

sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

print(sf_train.head())

# Compile the model with Cross Entropy Loss# Compile the model with Cross Entropy Loss

#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])# Compile the model with Cross Entropy Loss

#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])





# Pandas read CSV

#sf_train = pd.read_csv('data.csv')



# Correlation Matrix for target

#corr_matrix = sf_train.corr()

#print(corr_matrix['type'])



# Drop unnecessary columns

#sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

#print(sf_train.head())



# Pandas read Validation CSV

sf_val = pd.read_csv('/kaggle/input/dota-heroes/p5_val_data.csv')



# Drop unnecessary columns

sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)



# Get Pandas array value (Convert to NumPy array)

train_data = sf_train.values

val_data = sf_val.values



# Use columns 2 to last as Input

train_x = train_data[:,2:]

val_x = val_data[:,2:]



# Use columns 1 as Output/Target (One-Hot Encoding)

train_y = to_categorical( train_data[:,1] )

val_y = to_categorical( val_data[:,1] )



# Create Network

inputs = Input(shape=(16,))

h_layer = Dense(10, activation='sigmoid')(inputs)



# Softmax Activation for Multiclass Classification

outputs = Dense(3, activation='softmax')(h_layer)



model = Model(inputs=inputs, outputs=outputs)



# Optimizer / Update Rule

sgd = SGD(lr=0.001)



# Compile the model with Cross Entropy Loss

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model and use validation data

model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=1, validation_data=(val_x, val_y))

model.save_weights('weights.h5')



# Predict all Validation data

predict = model.predict(val_x)



# Visualize Prediction

df = pd.DataFrame(predict)

df.columns = [ 'Strength', 'Agility', 'Intelligent' ]

df.index = val_data[:,0]

print(df)
data_plt = predict

for data_last in data_plt:

    plt.plot(df.columns, data_last)

    plt.grid()

    plt.show()