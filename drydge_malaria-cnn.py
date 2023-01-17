# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

%matplotlib inline

from tensorflow.keras import datasets, layers, models



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

        

        

image_root = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/'

# image_root = 'cell_images/'

infect = 'Parasitized'

uninfect = 'Uninfected'

# Any results you write to the current directory are saved as output.
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import matplotlib.pyplot as plt

from PIL import Image

import cv2

img=cv2.imread(image_root+'Parasitized/C183P144NThinF_IMG_20151201_224458_cell_121.png')

imgplot = plt.imshow(img)

image_array = Image.fromarray(img , 'RGB')

resize_img = image_array.resize((100 , 100))

plt.imshow(resize_img)

plt.show()



img=cv2.imread(image_root+'Uninfected/C3thin_original_IMG_20150608_163047_cell_168.png')

imgplot = plt.imshow(img)

image_array = Image.fromarray(img , 'RGB')

resize_img = image_array.resize((100 , 100))

plt.imshow(resize_img)

plt.show()
data = []

labels = []

size = 50

import gc

gc.collect()

cols = ['img','label']

for dirname, _, filenames in os.walk(image_root):

    for filename in filenames:

        if(filename.split('.')[-1] != 'png'): continue

        dirname = dirname.replace('\\','/')

        label = dirname.split('/')[-1]

        img = cv2.imread(os.path.join(dirname, filename))



        image_array = Image.fromarray(img , 'RGB')

        resize_img = image_array.resize((size , size))

        data.append(np.array(resize_img))

        labels.append(np.array(label))

    


cells = np.array(data)

labels = np.array(labels)



np.save('Cells' , cells)

np.save('Labels' , labels)
n = np.arange(cells.shape[0])

np.random.shuffle(n)

cells = cells[n]

labels = labels[n]

labels_id = np.where(labels==infect, 1,labels)

labels_id = np.where(labels_id==uninfect, 0,labels_id)

labels_id=labels_id.astype(int)
labels_id
print(cells[0].shape)

print(len(cells))

idx = int(len(cells)*80/100)

x_train = cells[:idx]

y_train = labels_id[:idx]

x_test = cells[idx:]

y_test = labels_id[idx:]



print("train:",len(x_train))

print("test:",len(x_test))
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(x_train, y_train, epochs=10, 

                    validation_data=(x_test, y_test))