# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Activations

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

from keras.utils.vis_utils import model_to_dot

import matplotlib.pyplot as plt

import cv2

import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix as cm

from random import randint

from IPython.display import SVG

import matplotlib.gridspec as gridspec
def image_read(directory):

    images = list()

    labels = list()

    label = 0

    for label_name in os.listdir(directory):

        if label_name == 'glacier':

            label = 2

        elif label_name == 'sea':

            label = 4

        elif label_name == 'street':

            label = 5

        elif label_name == 'mountain':

            label = 3

        elif label_name == 'buildings':

            label = 0

        elif label_name == 'forest':

            label = 1

        

        for image_name in os.listdir(directory + r'/' + label_name):

            image = cv2.imread(directory + r'/' + label_name + r'/' + image_name)

            image = cv2.resize(image, (150, 150))

            

            images.append(image)

            labels.append(label)

        

    return shuffle(images, labels, random_state = 817328462)
def get_classlabel(class_code):

    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    return labels[class_code]
images, labels = image_read('/kaggle/input/intel-image-classification/seg_train/seg_train')
Images = np.array(images) #converting the list of images to numpy array.

Labels = np.array(labels)
print("Shape of the training images: ", Images.shape)

print("Shape of the trainig labels: ", Labels.shape)
a, b = plt.subplots(5, 5)

a.subplots_adjust(0, 0, 3, 3)

for i in range(0, 5, 1):

    for j in range(0, 5, 1):

        random_number = randint(0, len(Images))

        b[i,j].imshow(Images[random_number])

        b[i,j].set_title(get_classlabel(Labels[random_number]))

        b[i,j].axis('off')

        
model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Flatten())

model.add(Layers.Dense(180,activation='relu'))

model.add(Layers.Dense(100,activation='relu'))

model.add(Layers.Dense(50,activation='relu'))

model.add(Layers.Dropout(rate=0.5))

model.add(Layers.Dense(6,activation='softmax'))
model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
Utils.plot_model(model,to_file='model.png',show_shapes=True)
trained = model.fit(Images,Labels,epochs=35,validation_split=0.30)
plt.plot(trained.history['acc'])

plt.plot(trained.history['val_acc'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Test'], loc = 'upper left')

plt.show()



plt.plot(trained.history['loss'])

plt.plot(trained.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Test'], loc = 'upper left')

plt.show()
test_images,test_labels = image_read('/kaggle/input/intel-image-classification/seg_test/seg_test/')

test_images = np.array(test_images)

test_labels = np.array(test_labels)

model.evaluate(test_images,test_labels, verbose=1)
pred_images,no_labels = image_read('../input/intel-image-classification/seg_pred/')

pred_images = np.array(pred_images)

pred_images.shape
fig = plt.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)



for i in range(25):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    rnd_number = randint(0,len(pred_images))

    pred_image = np.array([pred_images[rnd_number]])

    pred_class = get_classlabel(model.predict_classes(pred_image)[0])

    pred_prob = model.predict(pred_image).reshape(6)

    for j in range(2):

        if (j%2) == 0:

            ax = plt.Subplot(fig, inner[j])

            ax.imshow(pred_image[0])

            ax.set_title(pred_class)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plt.Subplot(fig, inner[j])

            ax.bar([0,1,2,3,4,5],pred_prob)

            fig.add_subplot(ax)



            fig.show()
