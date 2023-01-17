# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/fruits/fruits-360_dataset/fruits-360/"))



# Any results you write to the current directory are saved as output.
import os

import gc

import psutil

import pandas as pd

import numpy  as np
import matplotlib.pyplot as plt 
import os

PATH = "../input/fruits/fruits-360_dataset/fruits-360/Training/"

PATH1 = "../input/fruits/fruits-360_dataset/fruits-360/Test/"
#Uncomment to check tha files in the directory

##os.listdir(PATH1)
data_dir_list = os.listdir(PATH)

data_dir_list1 = os.listdir(PATH1)
img_rows=64

img_cols=64

num_channel=3 



num_epoch=3

batch_size=100

img_data_list=[]

classes_names_list=[]

training_label = []
process = psutil.Process(os.getpid())

print('start', process.memory_info().rss)
import cv2



for dataset in data_dir_list:

    classes_names_list.append(dataset) 

    #print ('Loading images from {} folder\n'.format(dataset)) 

    img_list=os.listdir(PATH+'/'+ dataset)

    img_path = dataset

    for img in img_list:

        input_img=cv2.imread(PATH + '/'+ dataset + '/'+ img )

        input_img_resize=cv2.resize(input_img,(img_rows, img_cols))

        img_data_list.append(input_img_resize)

        training_label.append(img_path)
img_data_list1=[]

classes_names_list1=[]

Test_label = []
import cv2



for dataset in data_dir_list1:

    classes_names_list1.append(dataset) 

    #print ('Loading images from {} folder\n'.format(dataset)) 

    img_list1=os.listdir(PATH1+'/'+ dataset)

    img_path1 = dataset

    for img in img_list1:

        input_img1=cv2.imread(PATH1 + '/'+ dataset + '/'+ img )

        input_img_resize1=cv2.resize(input_img1,(img_rows, img_cols))

        img_data_list1.append(input_img_resize1)

        Test_label.append(img_path1)
num_classes = len(classes_names_list)

print(num_classes)

training_label = np.array(training_label)

Test_label = np.array(Test_label)
import numpy as np

img_data = np.array(img_data_list)

img_data = img_data.astype('float32')

img_data /= 255
img_data1 = np.array(img_data_list1)

img_data1 = img_data1.astype('float32')

img_data1 /= 255
del img_data_list1

del img_data_list
gc.collect()
print (img_data.shape)
plt.imshow(img_data[1])
label_to_id = {v : k for k, v in enumerate(np.unique(training_label))}

id_to_label = {v : k for k, v in label_to_id.items()}

label_to_id1 = {v : k for k, v in enumerate(np.unique(Test_label))}

id_to_label1 = {v : k for k, v in label_to_id1.items()}
## Uncomment to check the list of labels

#id_to_label1
num_of_samples = img_data.shape[0]

input_shape = img_data[0].shape

Test_num_of_samples = img_data1.shape[0]

Test_input_shape = img_data1[0].shape
classes = np.ones((num_of_samples,), dtype='int64')

Test_classes = np.ones((Test_num_of_samples,), dtype='int64')

classes = np.array([label_to_id[i] for i in training_label])

Test_classes = np.array([label_to_id1[i] for i in Test_label])
len(Test_label)
len(training_label)
from keras.utils import to_categorical



classes = to_categorical(classes, num_classes)

Test_classes = to_categorical(Test_classes, num_classes)
from sklearn.utils import shuffle



X, Y = shuffle(img_data, classes, random_state=2)

X_test,Y_test = shuffle(img_data1,Test_classes,random_state=4)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization

from keras.layers import Conv2D, MaxPooling2D
model = Sequential()



model.add(Conv2D(16, (3, 3), input_shape=input_shape,kernel_initializer="glorot_uniform" ))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Conv2D(16, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(32, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Conv2D(32, (3, 3),kernel_initializer='glorot_uniform'))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))





model.add(Conv2D(64, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Conv2D(64, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64,kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(32,kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(16,kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()
history = model.fit(X_train, y_train,validation_data=(X_val, y_val), epochs=100, batch_size=128, verbose=0)
import matplotlib.pyplot as plt





# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.evaluate(X_test,Y_test)
model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=input_shape,kernel_initializer="glorot_uniform" ))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Conv2D(64, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Conv2D(128, (3, 3),kernel_initializer='glorot_uniform'))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Conv2D(256, (3, 3),kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(600,kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(300,kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(150,kernel_initializer="glorot_uniform"))

model.add(BatchNormalization())

model.add(Activation(activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

model.summary()
history = model.fit(X_train, y_train,validation_data=(X_val, y_val), epochs=5, batch_size=128, verbose=1)
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.evaluate(X_test,Y_test)
from IPython.display import Image

Image(filename='../input/cnn-png/CNN.png')
from keras import models

from keras.callbacks import ModelCheckpoint

import glob

import matplotlib

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import imageio as im
layer_outputs = [layer.output for layer in model.layers[:20]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
import numpy as np

img_tensor = np.array(img_data[:15])

img_tensor = img_tensor.astype('float32')

img_tensor /= 255
print(img_tensor.shape)
activations = activation_model.predict(img_tensor)
layer_names = []

for layer in model.layers[:20]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')