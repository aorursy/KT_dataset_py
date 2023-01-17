import matplotlib.pyplot as plt 

import numpy as np

import os

import cv2

import IPython

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.model_selection import KFold

from PIL import Image

from keras import optimizers

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
print(os.listdir('../input'))
from PIL import Image

train_X=[]

test_X=[]

train=[]

test=[]

size=50

for root, dirs, files in os.walk("../input/fruit12345/fruits-360/Training"):

    for name in dirs:

        for filename in os.listdir(os.path.join(root, name)):

            image=Image.open( os.path.join(root, name) + "/"+filename)

            img_resized = np.array(image.resize((size,size)))

            train_X.append(img_resized)

            train.append(name)

            

for root, dirs, files in os.walk("../input/fruit12345/fruits-360/Test"):

    for name in dirs:

        for filename in os.listdir(os.path.join(root, name)):

            image=Image.open( os.path.join(root, name) + "/"+filename)

            img_resized = np.array(image.resize((size,size)))

            test_X.append(np.array(img_resized))

            test.append(name)

            

train_X=np.array(train_X)

test_X=np.array(test_X)
train_X,train=shuffle(train_X,train,random_state=44)

test_X,test=shuffle(test_X,test,random_state=44)
test=np.array(test)

train=np.array(train)



hot = OneHotEncoder()

train_y=train.reshape(len(train), 1)

train_y = hot.fit_transform(train_y).toarray()

test_y=test.reshape(len(test), 1)

test_y = hot.transform(test_y).toarray()

train_X=train_X/255
for k in range(10):

    i=np.random.randint(len(train))

    plt.imshow(train_X[i,:,:,:])

    plt.show()

    print("Reference :",train[i])
size=train_X.shape[1]



input_ = tf.keras.layers.Input((size, size, 3))

conv1 = tf.keras.layers.Conv2D(32, (7, 7),strides=(1, 1),padding="valid", activation="relu")(input_)

mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = tf.keras.layers.Conv2D(64, (3, 3),strides=(1, 1),padding="valid", activation="relu")(mp1)

mp2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = tf.keras.layers.Conv2D(128, (3,3),strides=(1, 1),padding="valid", activation="relu")(mp2)

mp3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

dr1 = tf.keras.layers.Dropout(0.2)(mp3)

conv4 = tf.keras.layers.Conv2D(256, (1,1),strides=(1, 1),padding="valid", activation="relu")(dr1)

mp4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

dr2 = tf.keras.layers.Dropout(0.2)(mp4)

fl = tf.keras.layers.Flatten()(dr2)

output = tf.keras.layers.Dense(8, activation="softmax")(fl)



model = tf.keras.Model(inputs = input_, outputs = output)



rmsprop = optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
tf.keras.utils.plot_model(model)
model.fit(train_X, train_y,batch_size = 32, epochs=30,verbose=0 )
scores = model.evaluate(test_X, test_y, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
for k in range(10):

    i=np.random.randint(len(test))

    plt.imshow(test_X[i,:,:,:])

    plt.show()

    prediction=model.predict(test_X[i:i+1,:,:,:])

    prediction=hot.inverse_transform(prediction)

    print("Prediction: ",prediction)

    print("Reference : ",test[i])
size=train_X.shape[1]



input_ = tf.keras.layers.Input((size, size, 3))

conv1 = tf.keras.layers.Conv2D(32, (7, 7),strides=(1, 1),padding="valid", activation="relu")(input_)

mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = tf.keras.layers.Conv2D(64, (3, 3),strides=(1, 1),padding="valid", activation="relu")(mp1)

mp2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

dr1 = tf.keras.layers.Dropout(0.2)(mp2)

fl = tf.keras.layers.Flatten()(dr1)

output = tf.keras.layers.Dense(8, activation="softmax")(fl)



model = tf.keras.Model(inputs = input_, outputs = output)



rmsprop = optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
model.fit(train_X, train_y,batch_size = 32, epochs=30,verbose=0 )
scores = model.evaluate(test_X, test_y, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  
for k in range(10):

    i=np.random.randint(len(test))

    plt.imshow(test_X[i,:,:,:])

    plt.show()

    prediction=model.predict(test_X[i:i+1,:,:,:])

    prediction=hot.inverse_transform(prediction)

    print("Prediction: ",prediction)

    print("Reference : ",test[i])