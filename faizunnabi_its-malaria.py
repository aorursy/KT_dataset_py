import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import os

import random

print(os.listdir("../input"))

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

#from keras.optimizers import SGD

%matplotlib inline
parasitized_cells_data = os.listdir('../input/cell_images/cell_images/Parasitized/')
unifected_cells_data = os.listdir('../input/cell_images/cell_images/Uninfected')
plt.bar(x=['Parasitized Cells','Unifected Cells'],height=[len(parasitized_cells_data),len(unifected_cells_data)])

plt.grid(alpha=0.2)
sample = random.sample(parasitized_cells_data,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('../input/cell_images/cell_images/Parasitized/'+sample[i])

    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    ax[i//3,i%3].imshow(im,interpolation="bicubic")

    ax[i//3,i%3].axis('off')

f.suptitle('Parasitized Cells')

plt.show()
sample = random.sample(unifected_cells_data,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('../input/cell_images/cell_images/Uninfected/'+sample[i])

    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    ax[i//3,i%3].imshow(im,interpolation="bicubic")

    ax[i//3,i%3].axis('off')

f.suptitle('Uninfected Cells')

plt.show()
plt.figure(figsize=(16,5))

plt.grid(alpha=0.2)

plt.xlim([0, 256])



plt.subplot(1,2,1)

plt.grid(alpha=0.2)

sample1 = random.sample(parasitized_cells_data,1)

im = cv2.imread('../input/cell_images/cell_images/Parasitized/'+sample1[0])

im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

channels = cv2.split(im)

colors = ("b", "g", "r") 

for(channel, c) in zip(channels, colors):

    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])

    plt.plot(histogram, color = c)

plt.xlabel("Color Distribution")

plt.ylabel("Pixels")

plt.title('Infected Cells')



plt.subplot(1,2,2)

plt.grid(alpha=0.2)

sample2 = random.sample(unifected_cells_data,1)

im = cv2.imread('../input/cell_images/cell_images/Uninfected/'+sample2[0])

im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

channels = cv2.split(im)

colors = ("b", "g", "r") 

for(channel, c) in zip(channels, colors):

    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])

    plt.plot(histogram, color = c)

plt.xlabel("Color Distribution")

plt.ylabel("Pixels")

plt.title('Uninfected Cells')





plt.show()
image_height = 80

image_width = 80

batch_size = 32

no_of_epochs  = 20
image_data = []

labels = []



for p in parasitized_cells_data:

    try:

        im = cv2.imread('../input/cell_images/cell_images/Parasitized/'+p)

        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

        im = cv2.resize(im,(image_width,image_height))

        image_data.append(im)

        labels.append(1)

    except:

        print("Error in loading "+p)
for u in unifected_cells_data:

    try:

        im = cv2.imread('../input/cell_images/cell_images/Uninfected/'+u)

        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

        im = cv2.resize(im,(image_width,image_height))

        image_data.append(im)

        labels.append(0)

    except:

        print("Error in loading "+u)
combined_data = list(zip(image_data, labels))

random.shuffle(combined_data)

image_data[:], labels[:] = zip(*combined_data)

image_data = np.array(image_data)

labels = np.array(labels)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.3, random_state=101)
train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=15,

                                   shear_range=0.2,

                                   zoom_range=0.2

                                   )



test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen.fit(X_train)

test_datagen.fit(X_test)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model = Sequential()

model.add(Conv2D(16,(3,3),padding= 'same',input_shape=(image_height,image_width,3),activation='relu'))

model.add(Conv2D(16,(3,3),padding= 'same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),padding= 'same',activation='relu'))

model.add(Conv2D(32,(3,3),padding= 'same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding= 'same',activation='relu'))

model.add(Conv2D(64,(3,3),padding= 'same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=500,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',

                                         factor=0.1,

                                         patience=2,

                                         cooldown=2,

                                         min_lr=0.00001,

                                         verbose=1)



callbacks = [reduce_learning_rate]
history = model.fit_generator(train_datagen.flow(X_train, y_train),

                    steps_per_epoch=len(X_train)//batch_size,

                    epochs=no_of_epochs,

                    validation_data=test_datagen.flow(X_test, y_test),

                    validation_steps=len(X_test)//batch_size,

                    callbacks=callbacks

                   )
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

fig = plt.figure(figsize=(16,9))



plt.subplot(1, 2, 1)

plt.plot(epochs, acc, 'g', label='Training Accuracy')

plt.plot(epochs, loss, 'r', label='Training loss')

plt.grid(axis='both')

plt.title('Training accuracy & loss')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs, val_acc, 'g', label='Validation acc')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.grid(axis='both')

plt.title('validation accuracy & loss')

plt.legend()