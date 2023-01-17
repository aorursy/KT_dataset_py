import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
from time import time
#Read train and test csv
train = pd.read_csv("../input/train.csv") #label,pixel0,pixel1,...,pixel783
test_x = pd.read_csv("../input/test.csv") #pixel0,pixel1,...,pixel783 (No Label)
train_y=train['label'] #Our target only the labels 
train_x=train.drop('label','columns') #Our source only the pixels, we drop the column 'label'
#Separate the first 2000 for training, the remain for validation
train_x,dev_x=train_x[2000:],train_x[:2000] 
train_y,dev_y=train_y[2000:],train_y[:2000]
#Data shape
print(train_x.shape,train_y.shape,dev_x.shape,dev_y.shape)
#Pandas to numpy
train_x=train_x.values
train_y=train_y.values
dev_x=dev_x.values
dev_y=dev_y.values
#Transform targets [0,9,...,7] to [[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],...,[0,0,0,0,0,0,0,1,0,0]]
train_y = tf.keras.utils.to_categorical(train_y, 10)
dev_y = tf.keras.utils.to_categorical(dev_y, 10)
#Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape((28,28,1), input_shape=(784,))) #Reshape input [Pixel0,...,Pixel783] to [Pixel0,.......,Pixel27]
                                                                                                        # [......................]
                                                                                                        # [......................]
                                                                                                        # [Pixel256,...,Pixel783]    
model.add(tf.keras.layers.BatchNormalization()) #BachNorm
model.add(tf.keras.layers.Activation("relu"))   #Relu Activation

model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2,2))) #2D Convolution Layer
model.add(tf.keras.layers.BatchNormalization()) #BachNorm
model.add(tf.keras.layers.Activation("relu")) #Relu Activation

model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2,2))) #2D Convolution Layer
model.add(tf.keras.layers.BatchNormalization()) #BachNorm
model.add(tf.keras.layers.Activation("relu")) #Relu Activation

model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2,2))) #2D Convolution Layer
model.add(tf.keras.layers.Dropout(0.5)) #Dropout
model.add(tf.keras.layers.Flatten()) #Flatten the 'squared' matrix to a (1,784) vector 
model.add(tf.keras.layers.Dense(128,activation='relu')) #Dense Layer with relu activation
model.add(tf.keras.layers.Dense(10, activation='softmax')) #Dense Layer with softmax activation so it can predict one of the 10 Labels

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())
#Train the model
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(train_x, train_y,
          batch_size=1000,
          epochs=10,
          verbose=1,
          validation_data=(dev_x, dev_y),
          callbacks=[tensorboard])
#Predict on the test set
prediction=model.predict(test_x)
#Save results
final=pd.DataFrame(np.array(prediction.argmax(axis=1)),columns=['Label'])
final.index += 1 
final.index.names = ['ImageId']
final.to_csv("result.csv")
