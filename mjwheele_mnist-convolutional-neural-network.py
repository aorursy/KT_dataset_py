import numpy as np 

import pandas as pd 

import csv         

import matplotlib.pyplot as plt #This lets us display that first image!

import random



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from keras import initializers

from keras import regularizers

from keras import layers



#Tool for splitting data

from sklearn.model_selection import train_test_split



#from IPython.display import Image, display

import os  

print(os.listdir("../input"))
#Establish File Paths

train_path = "../input/train.csv"

test_path =  "../input/test.csv"

samp_sub = "../input/sample_submission.csv"



#Read train csv into a numpy array

reader = csv.reader(open(train_path), delimiter=",")   #reads our csv file in

temp = list(reader) #turns this csv into a list

#print(temp[1:])

train_data = np.array(temp[1:]).astype("float")            #turns this lsit into a numpy array

x = train_data[0:,1:]                            #gets all but the first column of the array, this is 784 pixels each for 60K sample images

y = train_data[0:,0]                             #This is the number that those pixels resemble

x /= 255                                         #makes pixes whiteness on a scale of 0-1 instead of 0-255



#Read test csv into a numpy array, mostly same as previous segment

reader = csv.reader(open(test_path), delimiter=",")

temp = list(reader)

real_x = np.array(temp[1:]).astype("float") 

real_x /= 255  #makes pixes whiteness on a scale of 0-1 instead of 0-255



# Split into validation and training data

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size = 0.2, random_state=1337)





#declare dimensions of depth for input images

real_x = real_x.reshape(real_x.shape[0], 28, 28, 1)

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)

test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)





print("complete")
print(train_x.shape)

print(train_y.shape)
#Create Neural Network Model

drop_out = 0.25



#Define Model Type

model = keras.models.Sequential() 

model.add(tf.keras.layers.Conv2D(32, kernel_size=5, padding = "valid",  activation='relu'))

model.add(tf.keras.layers.Dropout(drop_out))

model.add(tf.keras.layers.BatchNormalization(momentum=0.8))



model.add(tf.keras.layers.Conv2D(64, kernel_size=4, padding = "valid",  activation='relu'))

model.add(tf.keras.layers.Dropout(drop_out))

model.add(tf.keras.layers.BatchNormalization(momentum=0.8))



model.add(tf.keras.layers.MaxPool2D(pool_size = 2))



model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding = "valid",  activation='relu'))

model.add(tf.keras.layers.Dropout(drop_out))

model.add(tf.keras.layers.BatchNormalization(momentum=0.8))



# model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding = "valid",  activation='relu'))

# model.add(tf.keras.layers.Dropout(drop_out))

# model.add(tf.keras.layers.BatchNormalization(momentum=0.8))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation = 'relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))







#Compile Model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



print("complete")
model.fit(train_x, train_y, shuffle=True, batch_size=500, epochs=8, verbose=1)



test_loss, test_acc = model.evaluate(test_x, test_y)



#Prints out Loss and Accuracy from the evaluation above

print("... \n")

print("-----------------------------------")

print("LOSS: ", test_loss)  

print("ACCURACY: ",  test_acc)

print("-----------------------------------\n")
probabilities = model.predict(real_x)



predictions = []



for p in probabilities:

    predictions.append(np.argmax(p))



ids = []

for i in range (1,28001):

    ids.append(i)

    

# The lines below shows you how to save your data in the format needed to score it in the competition

output = pd.DataFrame({'ImageId': ids, 'Label': predictions})



output.to_csv('submission.csv', index=False)



print("Predictions Sucesfully Created")


