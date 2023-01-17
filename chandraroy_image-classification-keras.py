# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import keras.layers as Layers

import keras.activations as Activations

import keras.models as Models

import keras.optimizers as Optimizer

import keras.metrics as Metrics

import keras.utils as Utils



from keras.utils.vis_utils import model_to_dot



import os

import matplotlib.pyplot as plot

import cv2

import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix as CM

from random import randint

from IPython.display import SVG

import matplotlib.gridspec as gridspec

# use builtin in keras method to extract image from folder

#Here using a built in routine

def get_images(directory):

    Images = []

    Labels= [] # 0 for Building, 1 for forest, 2 for Glacier, 3 for Mountain, 4 for Sea, 5 for Street

    

    label = 0

    

    for labels in os.listdir(directory):

        if labels =='glacier':

            label=2

        elif labels=='sea':

            label=4

        elif labels=='buildings':

            label=0

        elif labels=='forest':

            label=1

        elif labels=='street':

            label=5

        elif labels =='mountain':

            label=3

        

        for image_file in os.listdir(directory+labels):

            image=cv2.imread(directory+labels+r'/'+image_file)

            image=cv2.resize(image,(150,150))

            Images.append(image)

            Labels.append(label)

    return shuffle(Images, Labels, random_state=8)

         

        
def get_classlabel(class_code):

    labels= {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    return labels[class_code]
#Images, Labels = get_images("../input/seg_train/seg_train/")

Images, Labels = get_images("../input/seg_train/seg_train/")



Images = np.array(Images)

Labels = np.array(Labels)
print(Images.shape)
print(Labels.shape)
#let us look at some of the images

f, ax = plot.subplots(5,5)

f.subplots_adjust(0,0,3,3)



for i in range(0,5,1):

    for j in range(0,5,1):

        rnd_number = randint(0, len(Images))

        ax[i,j].imshow(Images[rnd_number])

        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))

        ax[i,j].axis('off')
# Build a CNN model

model = Models.Sequential()



model.add(Layers.Conv2D(200, kernel_size=(3,3), activation='relu', input_shape=(150,150,3)))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Conv2D(140, kernel_size=(3,3), activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Flatten())

model.add(Layers.Dense(180, activation='relu'))

model.add(Layers.Dense(100, activation = 'relu'))

model.add(Layers.Dropout(rate=0.5))

model.add(Layers.Dense(6, activation='softmax'))
model.compile(optimizer=Optimizer.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
SVG(model_to_dot(model).create(prog='dot', format='svg'))

Utils.plot_model(model,to_file='model.png',show_shapes=True)
#train

#trained= model.fit(Images, Labels, epochs=35, validation_split=0.30)

trained= model.fit(Images, Labels, epochs=30, validation_split=0.30)
test_images,test_labels = get_images('../input/seg_test/seg_test/')

test_images = np.array(test_images)

test_labels = np.array(test_labels)

model.evaluate(test_images,test_labels, verbose=1)
pred_images,no_labels = get_images('../input/seg_pred/')

pred_images = np.array(pred_images)

pred_images.shape
predicted_label=[]

predicted_label.append(model.predict(pred_images))
print(predicted_label)