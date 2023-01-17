



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

##for dirname, _, filenames in os.walk('/kaggle/input'):

  ##  for filename in filenames:

    ##    print(os.path.join(dirname, filename))

import cv2



import tensorflow.keras.layers as Layers

#from Layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D

import tensorflow.keras.activations as Actications

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

from keras.utils.vis_utils import model_to_dot

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix as CM

from random import randint

from IPython.display import SVG

import matplotlib.gridspec as gridspec

# Any results you write to the current directory are saved as output.
def get_images(directory):

    Images = []

    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street

    label = 0

    

    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.

        if labels == 'glacier': #Folder contain Glacier Images get the '2' class label.

            label = 2

        elif labels == 'sea':

            label = 4

        elif labels == 'buildings':

            label = 0

        elif labels == 'forest':

            label = 1

        elif labels == 'street':

            label = 5

        elif labels == 'mountain':

            label = 3

        

        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder

            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)

            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)

            Images.append(image)

            Labels.append(label)

    

    return (Images,Labels)



def get_classlabel(class_code):

    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    

    return labels[class_code]
Images, Labels = get_images('../input/intel-image-classification/seg_train/seg_train/') #Extract the training images from the folders.



Images = np.array(Images) #converting the list of images to numpy array.

Labels = np.array(Labels)
model=Models.Sequential()



model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

#model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(2,2))

model.add(Layers.Dropout(0.5))

model.add(Layers.Conv2D(150,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(2,2))

#model.add(Layers.Conv2D(120,kernel_size=(3,3),activation='relu'))

#model.add(Layers.Dropout(0.5))

model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(2,2))

model.add(Layers.Conv2D(80,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

model.add(Layers.Dropout(0.5))

model.add(Layers.Flatten())

model.add(Layers.Dense(180, activation='relu'))



model.add(Layers.Dense(100, activation='relu'))

model.add(Layers.Dense(50, activation='relu'))

model.add(Layers.Dropout(0.5))

model.add(Layers.Dense(6, activation='softmax'))

model.summary()

model.compile(optimizer = Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
trained = model.fit(Images,Labels,epochs=35,validation_split=0.30)

test_images,test_labels = get_images('../input/intel-image-classification/seg_test/seg_test/')

test_images = np.array(test_images)

test_labels = np.array(test_labels)

model.evaluate(test_images,test_labels, verbose=1)
model2=Models.Sequential()



model2.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

model2.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))

model2.add(Layers.MaxPool2D(2,2))

model2.add(Layers.Dropout(0.5))

model2.add(Layers.Conv2D(150,kernel_size=(3,3),activation='relu'))

model2.add(Layers.MaxPool2D(2,2))

model2.add(Layers.Conv2D(120,kernel_size=(3,3),activation='relu'))

model2.add(Layers.Dropout(0.5))

model2.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))

model2.add(Layers.MaxPool2D(2,2))

model2.add(Layers.Conv2D(80,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

model2.add(Layers.Dropout(0.5))

model2.add(Layers.Flatten())

model2.add(Layers.Dense(180, activation='relu'))



model2.add(Layers.Dense(100, activation='relu'))

model2.add(Layers.Dense(30, activation='relu'))

model2.add(Layers.Dropout(0.5))

model2.add(Layers.Dense(6, activation='softmax'))

model2.summary()

model2.compile(optimizer = Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
trained2 = model2.fit(Images,Labels,epochs=35,validation_split=0.30)

#test_images,test_labels = get_images('../input/intel-image-classification/seg_test/seg_test/')

#test_images = np.array(test_images)

#test_labels = np.array(test_labels)

model2.evaluate(test_images,test_labels, verbose=1)
pred_images,no_labels = get_images('../input/intel-image-classification/seg_pred/')

pred_images = np.array(pred_images)

pred_images.shape



fig = plt.figure(figsize=(60, 60))

outer = gridspec.GridSpec(4, 4, wspace=0.2, hspace=0.3)



for i in range(16):

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    rnd_number = randint(0,len(pred_images))

    pred_image = np.array([pred_images[rnd_number]])

    pred_class = get_classlabel(model2.predict_classes(pred_image)[0])

    pred_prob = model2.predict(pred_image).reshape(6)

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

            ax.bar([" Building","forest","glacier","mountain","Sea","Street"],pred_prob)

            fig.add_subplot(ax)





fig.show()