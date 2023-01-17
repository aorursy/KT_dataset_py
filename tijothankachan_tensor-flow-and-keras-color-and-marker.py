#importing necessary libraries

import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd
#load the data

train_df = pd.read_csv('../input/color-and-marker/train.csv')

# input data to be trained

x = np.column_stack((train_df.x.values, train_df.y.values))
# cocatinating output data to be trained

one_hot_color = pd.get_dummies(train_df.color).values

one_hot_marker = pd.get_dummies(train_df.marker).values



labels = np.concatenate((one_hot_color, one_hot_marker), axis=1)
# load test data

test_df = pd.read_csv('../input/color-and-marker/test.csv')

test_x = np.column_stack((test_df.x.values, test_df.y.values))



test_one_hot_color = pd.get_dummies(test_df.color).values

test_one_hot_marker = pd.get_dummies(test_df.marker).values



test_labels = np.concatenate((test_one_hot_color, test_one_hot_marker), axis=1)
model = keras.Sequential([

    keras.layers.Dense(64, input_shape=(2,), activation='relu'),    #passing number of neurons in hidden layer,output layer,input layer

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(9, activation='sigmoid')])



model.compile(optimizer='adam',                                     # compiling  

              loss=keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
#shuffling

np.random.RandomState(seed=42).shuffle(x)

np.random.RandomState(seed=42).shuffle(labels)
#fitting

model.fit(x, labels, batch_size=4, epochs=10)

# evaluation

print("EVALUATION")

model.evaluate(test_x, test_labels)





print("Prediction", np.round(model.predict(np.array([[0,3], [0,1], [-2, 1]]))))
# converting back to color and marker

print('prediction',np.round(model.predict(np.array([[0,1]]))))

a=np.round(model.predict(np.array([[0,1]])))

color_dict={0:'red', 1:'blue', 2:'green', 3:'teal', 4:'orange', 5:'purple', 6:'^', 7:'+', 8:'*'}

for i in range(0,9):

        if a[0][i]==1:

            print(color_dict[i], end =" ")