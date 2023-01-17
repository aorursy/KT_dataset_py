# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Deep Learning Model(based on Tensorflow,keras and CNN) for number recognition.ipynb

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
mnist = keras.datasets.mnist

#split dataset into training and testing dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# Normalize the dataset in range of 0 to 1
x_train = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)

# build owr cnn model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = x_train[0].shape))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

#Parameters for training model
model.compile(optimizer = 'adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train our model
model.fit(x_train,y_train,epochs = 3)

# Calculate the validation loss and validation accuracy
val_loss, val_acc = model.evaluate(x_test,y_test)
print("Validation loss for the model is {} and validation accuracy for the model is {}".format(val_loss,val_acc))
# Save model
model.save('epic_num_reader.model')

# Load model 
new_model = keras.models.load_model('epic_num_reader.model')
# Prediction
prediction = new_model.predict([x_test])
# print(prediction)
print(np.argmax(prediction[0]))

# Let's validate is the prediction is true or not.
plt.imshow(x_test[0])
plt.show()