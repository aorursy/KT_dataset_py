import numpy as np

import pickle

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
data = pickle.load(open("/kaggle/input/traffic-signs-preprocessed/data0.pickle","rb"))
X_test = data['x_test']

X_validation = data['x_validation']

Y_test = data['y_test']

Y_validation = data['y_validation']

X_train = data['x_train']

Y_train = data['y_train']

del data
def one_hottie(labels,C):

    One_hot_matrix = tf.one_hot(labels,C)

    return tf.keras.backend.eval(One_hot_matrix)

Y_train = one_hottie(Y_train, 43)

Y_validation = one_hottie(Y_validation, 43)

Y_test = one_hottie(Y_test, 43)

print ("Y_train shape: " + str(Y_train.shape))

print ("Y_test shape: " + str(Y_test.shape))

print ("Y_validation shape: " + str(Y_validation.shape))
example = np.transpose(X_train[8624],[1,2,0])

print(example.shape)

plt.imshow(example)
X_train = np.transpose(X_train/255.,[0,2,3,1])

X_test = np.transpose(X_test/255.,[0,2,3,1])

X_validation = np.transpose(X_validation/255.,[0,2,3,1])
# Implements the forward propagation for the model:

# CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, 1, activation='relu', input_shape=(32,32,3)),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),

    tf.keras.layers.MaxPool2D(padding = 'same',strides=2),

    tf.keras.layers.Conv2D(128, 5, activation='relu',padding="same"),

    tf.keras.layers.MaxPool2D(padding = 'same',strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(300, activation='relu'),

    tf.keras.layers.Dense(43, activation='softmax')

])

initial_learning_rate = 0.0001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate,

    decay_steps=80000,

    decay_rate=1,

    staircase=True)



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
result = model.fit(x=X_train,y=Y_train,batch_size=100,epochs=10,verbose=1,shuffle=False,initial_epoch=0)
plt.plot(result.history['accuracy'])

plt.plot(result.history['loss'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
valid = model.evaluate(X_test,Y_test,verbose=2)