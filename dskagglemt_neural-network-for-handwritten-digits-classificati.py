# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
len(X_train)
X_train[0].shape
X_train[0]
plt.matshow(X_train[0])
y_train[0]
X_train.shape

# 60000 --> Is the Number of Samples.

# 28 & 28 --> are each individual image.
# Flatten the array. Keeping the no. of samples as is.

X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_train_flattened.shape
X_train_flattened[0]
# Doing the same for test dataset.

X_test_flattened = X_test.reshape(len(X_test), 28*28)

X_test_flattened.shape
model = keras.Sequential([

    keras.layers.Dense(10, input_shape = (784,), activation = 'sigmoid')

])



model.compile(

    optimizer = 'adam',

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)



model.fit(X_train_flattened, y_train, epochs = 5)
# Scaled

X_train_scaled = X_train / 255

X_test_scaled = X_test / 255
X_train_scaled[0]

# See the data not.. it will range from 0 to 1.
# Flatten the array. Keeping the no. of samples as is.

X_train_scaled_flattened = X_train_scaled.reshape(len(X_train_scaled), 28*28)

X_test_scaled_flattened  = X_test_scaled.reshape(len(X_test_scaled), 28*28)
model.fit(X_train_scaled_flattened, y_train, epochs = 5)
# Evaluate the model

model.evaluate(X_test_scaled_flattened, y_test)
plt.matshow(X_test[0])
# Sample Prediction

y_pred = model.predict(X_test_scaled_flattened)

y_pred[0] # to get the predicted value of first array.
# Now get the max out of the array.

np.argmax(y_pred[0])
plt.matshow(X_test[1])
print(y_pred[1])

print('-'*10)

print(np.argmax(y_pred[1]))
# Lets get the labels for all predcited values.

y_pred_labels = [np.argmax(i) for i in y_pred]

y_pred_labels[:5]
# Confursion Matrix.

# Note, in below confusion matrix, we cannot direeclty pass y_pred in predictions, as y_pred is not real labels, where as y_test are representing real labels. Thus we have created y_pred_labels. 

cm = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_labels)

cm
# Visualize the COnfusion Matrix.

import seaborn as sn

plt.figure(figsize = (10,7))

sn.heatmap(cm, annot= True, fmt ='d')

plt.xlabel('Predicted')

plt.ylabel('Truth')
model = keras.Sequential([

    keras.layers.Dense(100, input_shape = (784,), activation = 'relu'),

    keras.layers.Dense(10, activation = 'sigmoid')

])



model.compile(

    optimizer = 'adam',

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)

model.fit(X_train_scaled_flattened, y_train, epochs = 5)
# Evaluate the model

model.evaluate(X_test_scaled_flattened, y_test)
# Sample Prediction

y_pred_1_Layer = model.predict(X_test_scaled_flattened)



# Lets get the labels for all predcited values.

y_pred_1_Layer_labels = [np.argmax(i) for i in y_pred_1_Layer]

# y_pred_1_Layer_labels[:5]



# Confursion Matrix.

cm_1_layer = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_1_Layer_labels)



#Visualize 

plt.figure(figsize = (10,7))

sn.heatmap(cm_1_layer, annot= True, fmt ='d')

plt.xlabel('Predicted')

plt.ylabel('Truth')
model_flatten = keras.Sequential([

    keras.layers.Flatten(input_shape = (28,28)),

    keras.layers.Dense(100, activation = 'relu'),

    keras.layers.Dense(10, activation = 'sigmoid')

])



model_flatten.compile(

    optimizer = 'adam',

    loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy']

)
model_flatten.fit(X_train_scaled, y_train, epochs = 5)
# Evaluate the model

model_flatten.evaluate(X_test_scaled, y_test)
# Sample Prediction

y_pred_f_Layer = model_flatten.predict(X_test_scaled)



# Lets get the labels for all predcited values.

y_pred_f_Layer_labels = [np.argmax(i) for i in y_pred_f_Layer]

# y_pred_f_Layer_labels[:5]



# Confursion Matrix.

cm_f_layer = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_f_Layer_labels)



#Visualize 

plt.figure(figsize = (10,7))

sn.heatmap(cm_f_layer, annot= True, fmt ='d')

plt.xlabel('Predicted')

plt.ylabel('Truth')