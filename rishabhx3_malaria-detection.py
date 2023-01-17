import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import keras

from keras import Sequential

from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator



from PIL import Image



import os

print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images"))
width = 128

height = 128
infected_folder = '../input/cell-images-for-detecting-malaria/cell_images/Parasitized/'

uninfected_folder  = '../input/cell-images-for-detecting-malaria/cell_images/Uninfected/'
print(len(os.listdir(infected_folder)))

print(len(os.listdir(uninfected_folder)))
# Infected cell image

rand_inf = np.random.randint(0,len(os.listdir(infected_folder)))

inf_pic = os.listdir(infected_folder)[rand_inf]



#Uninfected cell image

rand_uninf = np.random.randint(0,len(os.listdir(uninfected_folder)))

uninf_pic = os.listdir(uninfected_folder)[rand_uninf]



# Load the images

inf_load = Image.open(infected_folder+inf_pic)

uninf_load = Image.open(uninfected_folder+uninf_pic)
# Let's plt these images

f = plt.figure(figsize= (10,6))



a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(inf_load)

a1.set_title('Infected cell')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(uninf_load)

a2.set_title('Uninfected cell')
datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',

                                           target_size=(width,height),

                                           class_mode = 'binary',

                                           batch_size = 16,

                                           subset='training')
valDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',

                                           target_size=(width,height),

                                           class_mode = 'binary',

                                           batch_size = 16,

                                           subset='validation')
model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))



model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.3))



model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn_model = model.fit_generator(generator = trainDatagen,

                             steps_per_epoch = len(trainDatagen),

                              epochs =20,

                              validation_data = valDatagen,

                              validation_steps=len(valDatagen))
plt.plot(cnn_model.history['accuracy'])

plt.plot(cnn_model.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(cnn_model.history['val_loss'])

plt.plot(cnn_model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()