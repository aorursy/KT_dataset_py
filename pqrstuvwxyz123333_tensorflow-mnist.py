



import numpy as np 

import pandas as pd 

import tensorflow as tf

import tensorflow_datasets as tfds

mnist,mnist_info=tfds.load(name='mnist',with_info=True,as_supervised=True)
mnist_train,mnist_test=mnist['train'],mnist['test']
mnist_info
num_validation_samples=0.1*mnist_info.splits['train'].num_examples
num_validation_samples=tf.cast(num_validation_samples,tf.int64)
num_test_samples=tf.cast(0.1*mnist_info.splits['test'].num_examples,tf.int64)



def scale(image,label):

    image=tf.cast(image,tf.float32)

    image/=255.

    return image,label
scaled_train_and_validation_data=mnist_train.map(scale)

test_data=mnist_test.map(scale)
Buffer_size=10000

shuffle_train_and_validation_data=scaled_train_and_validation_data.shuffle(Buffer_size)

validation_data=shuffle_train_and_validation_data.take(num_validation_samples)

train_data=shuffle_train_and_validation_data.skip(num_validation_samples)

Batch_size=100

train_data=train_data.batch(Batch_size)

validation_data=validation_data.batch(num_validation_samples)

test_data=test_data.batch(num_test_samples)
validation_inputs,validation_targets=next(iter(validation_data))

test_inputs,test_targets=next(iter(test_data))

input_size=784

output_size=10

hidden_size=1000

model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28,1)),tf.keras.layers.Dense(hidden_size,activation='sigmoid'),tf.keras.layers.Dense(hidden_size,activation='relu'),tf.keras.layers.Dense(hidden_size,activation='tanh'),tf.keras.layers.Dense(hidden_size,activation='relu'),tf.keras.layers.Dense(output_size,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
num_epochs=5

model.fit(train_data,epochs=num_epochs,validation_data=(validation_inputs,validation_targets),validation_steps=1,verbose=1)
probablities=model.predict(test_inputs)
max=0

a=0

predicted=[]

for i in range(0,1000):

    max=probablities[i][0]

    a=0

    for j in range(0,10):

        if(probablities[i][j]>max):

            max=probablities[i][j]

            a=j

    predicted.append(a)
predicted
test_targets