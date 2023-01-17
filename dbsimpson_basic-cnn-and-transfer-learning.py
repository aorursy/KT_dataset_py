import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 #cv2 for image processing

import random

import os

import numpy as np #NumPy for array manipulation

import matplotlib.pyplot as plt #Matplotlib for visualizing the performance of the models

import matplotlib.image as mpimg

%matplotlib inline



import keras #Keras is a library for building neural networks on top of TensorFlow

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img

from keras.applications import InceptionResNetV2

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras import layers, models, optimizers



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
mask_dir = '../input/face-mask-detection-data/with_mask'

no_mask_dir = '../input/face-mask-detection-data/without_mask'

mask_img = [f'{mask_dir}/{i}' for i in os.listdir(mask_dir)]

no_mask_img = [f'{no_mask_dir}/{i}' for i in os.listdir(no_mask_dir)]
#Identify how many images we have in each group and total images

print("Total number of images with mask: " + str(len(mask_img)))

print("Total number of images without mask: " + str(len(no_mask_img)))

print("Total images: " + str(len(mask_img) + len(no_mask_img)))
#Split the mask and no mask into training and testing sets

tr_mask = mask_img[0:1499]

tr_no_mask = no_mask_img[0:1499]

test_mask = mask_img[1500:]

test_no_mask = no_mask_img[1500:]



#Combine the training and testing sets

train_img = tr_mask + tr_no_mask

test_img = test_mask + test_no_mask
#Define a function to resive and convert the images to the 3 channel BGR color image

#Also creates labels for with mask = 0 and without mask = 1 for classification use in neural network

def process_imgs(imgs, width=150, height=150):

    x = []

    y = []

    for i in imgs:

        x.append(cv2.resize(cv2.imread(i, cv2.IMREAD_COLOR), (width, height), interpolation=cv2.INTER_CUBIC))

        label = 1 if 'without' in i else 0

        y.append(label)

    return np.array(x), np.array(y)



tr_x, tr_y = process_imgs(train_img)

test_x, test_y = process_imgs(test_img)



# plot 5 images just to see the results of processing the images

plt.figure(figsize=(20, 10))

cols = 5

for i in range(cols):

    plt.subplot(5 / cols+1, cols, i+1) #keras

    plt.imshow(tr_x[i])
#Image data augmentation for use in TensorFlow with test and training sets

tr_data = ImageDataGenerator(rescale=1/255,

                            rotation_range=40,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True)



tr_gen = tr_data.flow(tr_x, tr_y, batch_size=32)

test_gen = tr_data.flow(test_x, test_y, batch_size = 32)
#Designing our first CNN for training

model = models.Sequential()

model.add(Conv2D(64, (1, 1), input_shape = (150,150,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (1, 1), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))
print(model.summary())
#Compiling and training our first CNN using RMSprop as an optimizer

batch_size = 32

epochs = 20

model.compile(loss='sparse_categorical_crossentropy',

             optimizer=optimizers.RMSprop(lr=2e-5),

             metrics=['acc'])

hist = model.fit(tr_gen, steps_per_epoch=tr_x.shape[0] // batch_size, epochs=epochs)
#Comparing the accuracy and loss of our first CNN on the test data

results = model.evaluate(test_gen, batch_size = 32)

print("Test loss and test accuracy: ", results)
#Graphing the loss and accuracy for our first CNN

epochs = list(range(1, len(hist.history['acc'])+1))

accuracy = hist.history['acc']

loss = hist.history['loss']





plt.subplot(2,1,1)

plt.plot(epochs, accuracy)

plt.title("CNN for Accuracy and Loss (Mask vs No Mask)")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.subplot(2,1,2)

plt.plot(epochs, loss)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
#Rebuilding the same model to compile using a different optimizer

model2 = models.Sequential()

model2.add(Conv2D(64, (1, 1), input_shape = (150,150,3), activation='relu'))

model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(128, (1, 1), activation='relu'))

model2.add(layers.Flatten())

model2.add(layers.Dense(256, activation='relu'))

model2.add(layers.Dense(2, activation='softmax'))
#Compiling and training our first CNN using ADAM as an optimizer

batch_size = 32

epochs = 20

model2.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['acc'])

hist2 = model2.fit(tr_gen, steps_per_epoch=tr_x.shape[0] // batch_size, epochs=epochs)
results2 = model2.evaluate(test_gen, batch_size = 32)

print("Test loss and test accuracy: ", results2)
epochs2 = list(range(1, len(hist2.history['acc'])+1))

accuracy2 = hist2.history['acc']

loss2 = hist2.history['loss']



plt.subplot(2,1,1)

plt.plot(epochs2, accuracy2)

plt.title("CNN for Accuracy and Loss (Mask vs No Mask)")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.subplot(2,1,2)

plt.plot(epochs2, loss2)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
#Loading pre-trained weights from ImageNet to save training time, thank you Transfer Learning

base=InceptionResNetV2(weights='imagenet',

                             include_top=False,

                             input_shape=(150, 150, 3))
model3 = models.Sequential()

model3.add(base)

model3.add(layers.Flatten())

model3.add(layers.Dense(256, activation='relu'))

model3.add(layers.Dense(2, activation='softmax'))

base.trainable = False
print(model3.summary())
batch_size = 32

epochs = 20

model3.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['acc'])

hist3 = model3.fit(tr_gen, steps_per_epoch=tr_x.shape[0] // batch_size, epochs=epochs)
#Comparing the accuracy and loss of our model relying on ImageNet model as the first layer

results3 = model3.evaluate(test_gen, batch_size = 32)

print("Test loss and test accuracy: ", results3)
#Graphing the loss and accuracy for our model using ImageNet model as the first layer

epochs3 = list(range(1, len(hist3.history['acc'])+1))

accuracy3 = hist3.history['acc']

loss3 = hist3.history['loss']





plt.subplot(2,1,1)

plt.plot(epochs3, accuracy3)

plt.title("ImageNet and our NN for Accuracy and Loss (Mask vs No Mask)")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.subplot(2,1,2)

plt.plot(epochs3, loss3)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()