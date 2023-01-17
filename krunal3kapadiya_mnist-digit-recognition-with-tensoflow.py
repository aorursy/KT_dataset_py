#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization library
from sklearn.model_selection import train_test_split # training and splitting data

import tensorflow as tf #tensorflow

import os
print(os.listdir("../input"))
#loading training and testing dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.columns
test_df.columns
# seperating labels and images [X_train = images, Y_train = numbers on respective image]
X_train = train_df.drop(labels = ["label"],axis = 1) # contains values of digits in 255 range
Y_train = train_df['label'] # contains digits
X_train = X_train.values.reshape(-1,28,28,1)/ 255 # reshaping arrays in tensors
# creating common method to display image
def displayImage(image):
    plt.imshow(image[:,:,0], cmap=plt.cm.binary)
    
def displayImageWithPredictedValue(image, prediction):
    print('Predicted output image is ', np.argmax(prediction))
    plt.imshow(image[:,:,0], cmap=plt.cm.binary)
# displaying first first value
displayImage(X_train[0])
model = tf.keras.models.Sequential() # creating Sequential model
model.add(tf.keras.layers.Flatten()) # flattening the input arrays
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # using relu activation function
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu)) # using relu activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # activation function to get number of output

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compiling model

model.fit(X_train, Y_train.values, epochs=5) # training model and fitting data
model.summary()
# splitting data to evalueate model
X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                              Y_train, 
                                              test_size=0.20,
                                              random_state=42,
                                              shuffle=True,
                                              stratify=Y_train)
val_loss , val_accuracy = model.evaluate(X_val, Y_val) # evaluating performance of the model
print(val_loss, val_accuracy)
predictions = model.predict([X_val])
displayImageWithPredictedValue(X_val[12], predictions[12])
test_df = test_df.values.reshape(-1,28,28,1)/255
predictions = model.predict([test_df])
displayImageWithPredictedValue(test_df[10], predictions[10])
# creating array of outputs, to add into submission.csv file
results = np.argmax(predictions,axis = 1)
# creating submission.csv file
submission = pd.DataFrame(data={'ImageId': (np.arange(len(predictions)) + 1), 'Label': results})
submission.to_csv('submission.csv', index=False)