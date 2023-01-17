#Import all the neccessary libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
#Augment the images
datagen = ImageDataGenerator(
rotation_range=15,
rescale=1/255,
zoom_range=0.1,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1,
validation_split=0.2)
#Train dataset
trainDatagen = datagen.flow_from_directory(directory='../input/asl-dataset/asl_dataset',
                                           target_size=(64, 64),
                                           class_mode = 'categorical',
                                           batch_size = 64,
                                           subset = 'training')
#Assign numbers to each category
trainDatagen.class_indices
#Validation dataset
valDatagen = datagen.flow_from_directory(directory='../input/asl-dataset/asl_dataset',
                                           target_size=(64, 64),
                                           class_mode = 'categorical',
                                           batch_size = 64,
                                           subset='validation')
#Define the CNN layers
model = Sequential() #I'm going to be creating a sequential model
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(64, 64, 3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy']) #Compile the model

model.summary() #Get the model's summary
from keras.callbacks import EarlyStopping,ReduceLROnPlateau #Import callback functions
earlystop=EarlyStopping(patience=10) #Monitor the performance. If it dips, then stop training
learning_rate_reduce=ReduceLROnPlateau(monitor='val_acc',min_lr=0.001) #Change learning rate if not performing good enough
callbacks=[earlystop,learning_rate_reduce]
#Start training the model with 15 epochs
history=model.fit_generator(trainDatagen,
                            epochs=25,
                            validation_data=valDatagen ,
                           )
#plot this model
plt.plot(history.history['accuracy'],color='black', label='Accuracy')
plt.plot(history.history['loss'],color='blue', label='Loss')
plt.plot(history.history['val_accuracy'],color='yellow', label='Validation Accuracy')
plt.plot(history.history['val_loss'],color='red', label='Validation loss')
plt.legend()
plt.show()
model.save('asl.h5')
