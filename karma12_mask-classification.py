import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

from sklearn.utils import shuffle

from random import randint



from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout

import tensorflow.keras.activations as Actications

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import RMSprop,Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

import matplotlib.image as mpimg
def getImages(directory):

    

    Images = list()

    Labels = list()

    

    label = 0

    

    for di in os.listdir(directory):

        if di == 'Mask':

            label = 0

        elif di == 'Non Mask':

            label = 1

        for image_file in os.listdir(os.path.join(directory,di)):

            img_path = os.path.join(directory,di)

            image = cv2.imread(os.path.join(img_path,image_file))

            image = cv2.resize(image,(224,224))

            

            Images.append(image)

            Labels.append(label)

    return shuffle(Images, Labels, random_state= 0)
Images, Labels = getImages('../input/covid-face-mask-detection-dataset/New Masks Dataset/Train')



train_Images = np.array(Images)

train_Labels = np.array(Labels)
print(f'Train images shape: {train_Images.shape}')

print(f'Train Labels shape: {train_Labels.shape}')
def get_label(code):

    label = {0:'Mask', 1:'Non Mask'}

    

    return label[code]
fig, ax = plt.subplots(5,5)

fig.subplots_adjust(0,0,3,3)



for i in range(0, 5, 1):

    for j in range(0, 5, 1):

        rand_number = randint(0, len(Images))

        ax[i, j].imshow(train_Images[rand_number])

        ax[i, j].set_title(get_label(train_Labels[rand_number]))

        ax[i, j].axis('off')
data_gen = ImageDataGenerator(

    rescale=1./255,

    zoom_range = 0.2,

    horizontal_flip = True,

    vertical_flip = True,

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    validation_split=0.2)
base_dir = '../input/covid-face-mask-detection-dataset/New Masks Dataset'

train_data_gen = data_gen.flow_from_directory(os.path.join(base_dir,'Train'),

                                              target_size=(224, 224),

                                             batch_size = 32,

                                             class_mode='binary')

validation_data_gen = data_gen.flow_from_directory(os.path.join(base_dir,'Validation'),

                                              target_size=(224, 224),

                                             batch_size = 32,

                                             class_mode='binary')                                         
model = Sequential([

    Conv2D(16, (3, 3), activation = 'relu', input_shape = (224, 224,3)),

    MaxPool2D(2, 2),

    Conv2D(32, (3,3), activation='relu'),

    MaxPool2D(2,2),

    Conv2D(64, (3,3), activation='relu'),

    MaxPool2D(2,2),

    Conv2D(128, (3,3), activation='relu'),

    MaxPool2D(2,2),

    Dropout(0.1),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1, activation='sigmoid')

    

])
model.summary()
model.compile(loss='binary_crossentropy',

              optimizer=Adam(),

              metrics=['accuracy'])

history = model.fit(train_data_gen, epochs= 50, validation_data = validation_data_gen,verbose = 2)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
test_data_gen = data_gen.flow_from_directory(os.path.join(base_dir,'Test'),

                                              target_size=(224, 224),

                                             batch_size = 32,

                                             class_mode='binary')   
STEP_SIZE_TEST=test_data_gen.n//test_data_gen.batch_size



pred=model.predict_generator(test_data_gen,

steps=STEP_SIZE_TEST,

verbose=1)
scores = model.evaluate_generator(test_data_gen,50)

print("Accuracy = ", scores[1])
model.evaluate(test_data_gen)
def test_mask(path):

    im = mpimg.imread(path)

    plt.imshow(im)

    img = image.load_img(path, target_size=(224, 224))

    y = image.img_to_array(img)

    y = np.expand_dims(y, axis=0)

    images = np.vstack([y])

    classes = model.predict_classes(images, batch_size=10)

    return get_label(classes[0][0])
test_mask('../input/covid-face-mask-detection-dataset/New Masks Dataset/Test/Mask/2114.jpeg')