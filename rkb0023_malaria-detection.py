import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images"))
# keras libraries
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix
infected_dir = "../input/cell-images-for-detecting-malaria/cell_images/Parasitized/"
uninfected_dir = "../input/cell-images-for-detecting-malaria/cell_images/Uninfected/"
# infected pic
rand_norm = np.random.randint(0, len(os.listdir(infected_dir)))
infected_pic = os.listdir(infected_dir)[rand_norm]
infected_pic_address = infected_dir+infected_pic

# uninfected pic
rand_norm = np.random.randint(0, len(os.listdir(uninfected_dir)))
uninfected_pic = os.listdir(uninfected_dir)[rand_norm]
uninfected_pic_address = uninfected_dir+uninfected_pic

# load the images
infected_load = Image.open(infected_pic_address)
uninfected_load = Image.open(uninfected_pic_address)

# plot
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(infected_load, cmap='gray')
a1.set_title('Infected')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(uninfected_load, cmap='gray')
a2.set_title('Uninfected')
width = 100
height = 100
# loading the images
datagen = ImageDataGenerator(rescale=1./255,
                            validation_split=0.15,
                            rotation_range=30,
                            zoom_range=0.15,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            fill_mode="nearest")


trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                           target_size=(width,height),
                                           shuffle=True,
                                           class_mode = 'categorical',
                                           batch_size = 32,
                                           subset='training')

valDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                           target_size=(width,height),
                                           class_mode = 'categorical',
                                           batch_size = 32,
                                           subset='validation')
### Create Model from scratch using CNN
model=Sequential()
model.add(Conv2D(filters=256,kernel_size=2,padding="same",activation="relu",input_shape=(width,height,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(2,activation="softmax"))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss',patience=2)
history = model.fit_generator(generator=trainDatagen,
                              steps_per_epoch=len(trainDatagen),
                              epochs=10,
                              validation_data=valDatagen,
                              validation_steps=len(valDatagen),
                              callbacks=[early_stop])
def plotLearningCurve(history,epochs=20):
    epochRange = range(1,epochs+1)
    plt.plot(epochRange,history.history['accuracy'])
    plt.plot(epochRange,history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

    plt.plot(epochRange,history.history['loss'])
    plt.plot(epochRange,history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

plotLearningCurve(history, epochs=len(history.history['accuracy']))
img=image.load_img('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png',
                   target_size=(100,100))
x=image.img_to_array(img)
x
x.shape
x=x/255
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape
model.predict(img_data)
pred = np.argmax(model.predict(img_data), axis=1)
pred
if(pred==1):
    print("Uninfected")
else:
    print("Infected")
