# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

from keras.preprocessing import image

import matplotlib.pyplot as plt
# table to store the data

from prettytable import PrettyTable

table = PrettyTable()

table.field_names=["layers",'train accuracy', 'cv accuracy', 'test accuracy']

print(table)
p = Path('../input/drive-download-20190326t061218z-001')

dirs = p.glob('*')

label_count = 0

label_dict = {}

# it will separate class names fom the dictionary and creates a dictionary with key respective to them

for folder in dirs:

    label = str(folder).split('/')[-1]

    label_dict[label] = label_count

    label_count += 1



print("There are",len(label_dict), "classes\n")





for x, y in label_dict.items():

  print(x, y)



image_data_per_class = []

image_labels_per_class = []

p = Path('../input/drive-download-20190326t061218z-001')

dirs = p.glob('*')

total = 0

for folder in dirs:

    label = str(folder).split('/')[-1]

    cnt = 0

    img_data = []

    label_data = []

    print(label)

    # read png file 

    for img_path in folder.glob('*.png'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

    

    # read jpg files

    for img_path in folder.glob('*.jpg'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

        

    print("There are", cnt, "images in", label)

    image_data_per_class.append(img_data)

    image_labels_per_class.append(label_data)

print('There are' ,total,"images")
print(len(image_data_per_class[0]))

print(len(image_data_per_class[0][0]))

print(len(image_data_per_class[0][0][0]))

print(len(image_data_per_class[0][0][0][0]))
print(len(image_labels_per_class))
# method to draw the images

def drawImg(img, label):

    plt.imshow(img)

    for key, value in label_dict.items(): 

         if label == value: 

                plt.title(key)

    plt.show()
# Visualization

import numpy as np

for i in range(0,11):

    x = np.array(image_data_per_class[i])

    y = np.array(image_labels_per_class[i])

    drawImg(x[0]/255.0, y[0])
plt.plot( [x for x in range(0,11)], [len(y) for y in image_labels_per_class],'*')

plt.grid()

plt.xlabel('classes')

plt.ylabel('number of elements')

plt.show()
#get the class label with maximum data points

max = -1

label = -1

for i in range(0,11):

    if max < len(image_data_per_class[i]):

        max = len(image_data_per_class[i])

        label = i

print("max number of data points is", max, 'and label is', label)
from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range = 10, width_shift_range=0.1,

                        height_shift_range = 0.1, shear_range = 0.15, zoom_range = 0.1, 

                        channel_shift_range=10, horizontal_flip = True)
image_data_per_class = []

image_labels_per_class = []

p = Path('../input/drive-download-20190326t061218z-001')

dirs = p.glob('*')

total = 0

for folder in dirs:

    label = str(folder).split('/')[-1]

    cnt = 0

    size = 413

    img_data = []

    label_data = []

    img = []

    for img_path in folder.glob('*.png'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

    

    

    for img_path in folder.glob('*.jpg'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

    

    if size - cnt > 0:

        aug_iter = gen.flow(np.expand_dims(img,0))

        aug_samples = [next(aug_iter)[0].astype(np.uint8) for i in range(413-cnt)]

        for i in range (413-cnt):

            label_data.append(label_dict[label])

        total += len(aug_samples)

        for sample in aug_samples:

            img_array = image.img_to_array(sample)

            img_data.append(img_array)

            

    image_data_per_class.append(img_data)

    image_labels_per_class.append(label_data)

print('There are' ,total,"images")
plt.plot( [x for x in range(0,11)], [len(y) for y in image_labels_per_class],'*')

plt.grid()

plt.xlabel('classes')

plt.ylabel('number of elements')

plt.show()
# displaying random images from the balanced dataset we get

for i in range(11):

    rand_idx = np.random.randint(413)

    x = np.array(image_data_per_class[i])

    y = np.array(image_labels_per_class[i])

    drawImg(x[rand_idx]/255.0, y[rand_idx])
xtrain = []

ytrain = []

for data in image_data_per_class:

    xs = np.array(data[0:259])

    for x in xs:

        xtrain.append(x)

xtrain = np.array(xtrain)

print(xtrain.shape)

for data in image_labels_per_class:

    ys = np.array(data[0:259])

    for y in ys:

        ytrain.append(y)

ytrain = np.array(ytrain)

print(ytrain.shape)
xtest = []

ytest = []

for data in image_data_per_class:

    xs = np.array(data[259:330])

    for x in xs:

        xtest.append(x)

xtest = np.array(xtest)

print(xtest.shape)

for data in image_labels_per_class:

    ys = np.array(data[259:330])

    for y in ys:

        ytest.append(y)

ytest = np.array(ytest)

print(ytest.shape)
xcv = []

ycv = []

for data in image_data_per_class:

    xs = np.array(data[330:413])

    for x in xs:

        xcv.append(x)

xcv = np.array(xcv)

print(xcv.shape)

for data in image_labels_per_class:

    ys = np.array(data[330:413])

    for y in ys:

        ycv.append(y)

ycv = np.array(ycv)

print(ycv.shape)
import keras

xtrain /= 255.0

xcv /= 255.0

xtest /= 255.0

ytrain = keras.utils.to_categorical(ytrain, 11)

ycv = keras.utils.to_categorical(ycv, 11)

ytest = keras.utils.to_categorical(ytest, 11)
from keras.models import load_model

from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

from keras.models import Sequential

from keras.optimizers import Adam, SGD

from keras.callbacks import ModelCheckpoint

from keras import applications



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(150,150,3)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.25))

model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(11, activation='softmax'))

model.summary()


model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history = model.fit(xtrain, ytrain,

          batch_size=128,

          epochs=20,

          verbose=1,

          validation_data=(xcv, ycv))
x = list(range(1,21))

vy = history.history['val_loss']

ty = history.history['loss']



plt.plot(x, ty, 'b', label= 'Train loss')

plt.plot(x, vy, 'r', label= 'Val loss')

plt.legend()

plt.xlabel("Number Of Epochs")

plt.ylabel("Categorical Cross  Entrophy loss")

plt.show()

score = model.evaluate(xtest, ytest, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

table.add_row(["3 layer", 0.9954, 0.9562,0.9910371318822023])


model1 = Sequential()

model1.add(Conv2D(32, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(150,150,3)))



model1.add(Conv2D(64, (5, 5), activation='relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))





model1.add(Conv2D(64, (5, 5), activation='relu'))

model1.add(Dropout(0.25))





model1.add(Conv2D(128, (5, 5), activation='relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))



model1.add(Conv2D(64, (5, 5), activation='relu'))





model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dropout(0.5))

model1.add(Dense(11, activation='softmax'))

model1.summary()


model1.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history = model1.fit(xtrain, ytrain,

          batch_size=128,

          epochs=20,

          verbose=1,

          validation_data=(xcv, ycv))


score = model1.evaluate(xtest, ytest, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



x = list(range(1,21))

vy = history.history['val_loss']

ty = history.history['loss']



plt.plot(x, ty, 'b', label= 'Train loss')

plt.plot(x, vy, 'r', label= 'Val loss')

plt.legend()

plt.xlabel("Number Of Epochs")

plt.ylabel("Categorical Cross  Entrophy loss")

plt.show()

table.add_row(["5 layer", 0.9881, 0.9573,0.9884763124199744])
print(table)