import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

import os
data = "/kaggle/input/Kannada-MNIST/"
train = pd.read_csv(data+"train.csv")

test = pd.read_csv(data+"Dig-MNIST.csv")

X = pd.read_csv(data+"test.csv")
X_train = train.iloc[:,1:].values

Y_train = train.iloc[:,0].values

X_test = test.iloc[:,1:].values

Y_test = test.iloc[:,0].values

ID = X.iloc[:,0]

X = X.iloc[:,1:].values
X_train = X_train.reshape([-1,28,28,1])/255.

X_test = X_test.reshape([-1,28,28,1])/255.

X = X.reshape([-1,28,28,1])/255.
X_train = np.append(X_train,X_test,axis=0)

Y_train = np.append(Y_train,Y_test,axis=0)
print(Y_train.shape)

print(Y_test.shape)

print(X_train.shape)

print(X_test.shape)


plt.imshow(X_train[9][:,:,0])

Y_train[8]
def one_hottie(labels,C):

    One_hot_matrix = tf.one_hot(labels,C)

    return tf.keras.backend.eval(One_hot_matrix)

Y_train = one_hottie(Y_train, 10)

Y_test = one_hottie(Y_test, 10)

print ("Y shape: " + str(Y_train.shape))
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, 1, activation=None,kernel_regularizer=tf.keras.regularizers.l2(0.001), 

                           input_shape=(28,28,1)),

    tf.keras.layers.BatchNormalization(axis=3),

    tf.keras.layers.LeakyReLU(0.1),

    tf.keras.layers.MaxPool2D(strides=2,padding="same"),

    

    tf.keras.layers.Conv2D(128, 5, activation=None,padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(axis=3),

    tf.keras.layers.LeakyReLU(0.1),

    tf.keras.layers.MaxPool2D(strides=2,padding="same"),

    

    tf.keras.layers.Conv2D(256, 3, activation=None,kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(axis=3),

    tf.keras.layers.LeakyReLU(0.1),

    tf.keras.layers.MaxPool2D(strides=2,padding="same"),

    

    tf.keras.layers.Conv2D(64, 3, activation=None,padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    tf.keras.layers.BatchNormalization(axis=3),

    tf.keras.layers.LeakyReLU(0.1),

    tf.keras.layers.MaxPool2D(strides=2,padding="same"),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(300,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=None),

    tf.keras.layers.BatchNormalization(axis=1),

    tf.keras.layers.LeakyReLU(0.1),

    tf.keras.layers.Dropout(0.6),

    tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.001) ,activation='softmax')

])

initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate,

    decay_steps=1968,

    decay_rate=0.1,

    staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy'])
result = model.fit(x=X_test,y=Y_test,batch_size=64,epochs=30,verbose=1,shuffle=True,initial_epoch=0,

                   validation_data=(X_train,Y_train))
result.history.keys()
plt.plot(result.history['accuracy'], label='train')

plt.plot(result.history['val_accuracy'], label='valid')

plt.legend(loc='upper left')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()

plt.plot(result.history['loss'], label='train')

plt.plot(result.history['val_loss'], label='test')

plt.legend(loc='upper right')

plt.title('Model Cost')

plt.ylabel('Cost')

plt.xlabel('Epoch')

plt.show()
valid = model.evaluate(X_test,Y_test,verbose=2)
Records = []
Records.append(valid)

Records
label = model.predict_classes(X)

print(label.shape)

print(label)
ID = pd.DataFrame(ID.values,columns=["id"])
ans = ID.join(pd.DataFrame(label,columns=["label"]))

ans.head(20)
ans.to_csv("submission.csv",index=False)