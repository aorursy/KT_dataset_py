import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from PIL import Image



from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img



import os
train_folder= '../input/chest-xray-pneumonia/chest_xray/train/'

val_folder = '../input/chest-xray-pneumonia/chest_xray/val/'

test_folder = '../input/chest-xray-pneumonia/chest_xray/test/'
train_n = train_folder+'NORMAL/'

train_p = train_folder+'PNEUMONIA/'
#Normal pic 

rand_norm = np.random.randint(0,len(os.listdir(train_n)))

norm_pic = os.listdir(train_n)[rand_norm]

norm_pic_address = train_n + norm_pic



#Pneumonia

rand_p = np.random.randint(0,len(os.listdir(train_p)))

pne_pic =  os.listdir(train_p)[rand_p]

pne_pic_address = train_p + pne_pic



#Let's plot these images

f = plt.figure(figsize= (10,6))



a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(Image.open(norm_pic_address))

a1.set_title('Normal')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(Image.open(pne_pic_address))

a2.set_title('Pneumonia')
cnn_model = Sequential()



cnn_model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))



cnn_model.add(Conv2D(32, (3, 3), activation="relu"))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))



cnn_model.add(Flatten())



cnn_model.add(Dense(activation = 'relu', units = 128))

cnn_model.add(Dense(activation = 'sigmoid', units = 1))
cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



train_set = train_datagen.flow_from_directory(train_folder,

                                              target_size = (64, 64),

                                              batch_size = 32,

                                              class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)



validation_generator = test_datagen.flow_from_directory(val_folder,

                                                        target_size=(64, 64),

                                                        batch_size=32,

                                                        class_mode='binary')



test_set = test_datagen.flow_from_directory(test_folder,

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')
cnn_model_his = cnn_model.fit_generator(train_set,

                              steps_per_epoch = 163,

                              epochs = 10,

                              validation_data = validation_generator,

                              validation_steps = 624)
test_acc = cnn_model.evaluate_generator(test_set,steps=624)

print('The testing accuracy is :',test_acc[1]*100, '%')
plt.plot(cnn_model_his.history['accuracy'])

plt.plot(cnn_model_his.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(cnn_model_his.history['val_loss'])

plt.plot(cnn_model_his.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()