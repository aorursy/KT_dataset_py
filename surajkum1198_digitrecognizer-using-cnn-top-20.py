import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataset=pd.read_csv('../input/digit-recognizer/train.csv')
dataset
X_train=np.array(dataset.drop(['label'],axis=1))

Y_train=np.array(dataset['label'])
X_train
Y_train
X_train=X_train.reshape(X_train.shape[0],28,28,1)

X_train=X_train.astype('float32')

X_train=X_train/255
plt.figure()

plt.imshow(X_train[0][:,:,0])
# CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1,

        

         )
import tensorflow as tf
X_train.shape
# model=tf.keras.Sequential([

#     tf.keras.layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu,input_shape=(28, 28, 1)),

#     tf.keras.layers.MaxPooling2D((2,2),strides=2),

#     tf.keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu),

#     tf.keras.layers.MaxPooling2D((2,2),strides=2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(512,activation=tf.nn.relu),

#     tf.keras.layers.Dense(10,activation=tf.nn.softmax)

# ])

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D((2,2),strides=2),



#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

#     tf.keras.layers.MaxPooling2D((2,2),strides=2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2),strides=2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D((2,2),strides=2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10,activation=tf.nn.softmax)

])
model.compile(optimizer="adam",

              loss="sparse_categorical_crossentropy",

              metrics=["accuracy"]

             )
model.fit_generator(datagen.flow(X_train,Y_train, batch_size=64),

        epochs = 60, steps_per_epoch = X_train.shape[0]//64)
test_df=pd.read_csv('../input/digit-recognizer/test.csv')
X_test=np.array(test_df)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test=X_test.astype('float32')
X_test/=255
predictions=model.predict(X_test)
predictions[0]
np.argmax(predictions[10])
plt.figure()

plt.imshow(X_test[10][:,:,0])
X_test.shape
predictions.shape
results=[]
for i in range(28000):

    results.append(np.argmax(predictions[i]))

    
results[4]
results=pd.Series(results,name="Label")
results
submission=pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)
submission
submission.to_csv('My_submissions6',index=False)
my_sub=pd.read_csv('My_submissions6')
my_sub