import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")



y_train = train["label"]

x_train = train.drop(labels = ["label"],axis = 1) 
x_train = x_train/255.0

test = test/255.0



x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=5)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)),

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'),

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

    tf.keras.layers.Dropout(0.25),

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation = "relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation= "softmax")

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2)  # randomly shift images vertically (fraction of total height)
batch_size = 80

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = 100, validation_data = (x_val,y_val),

                              verbose = 2, steps_per_epoch = 100)
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("nn_mnist.csv",index=False)
