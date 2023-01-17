# Disable warnings

import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



import os

print("Root: ", os.listdir("../input"))

print("Train: ", os.listdir("../input/seg_train/seg_train/"))



import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Actications

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

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

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
def loadData(path):

    

    # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street

    Images = []

    Labels = []  

    label = 0

    

    # We search in the main folder 

    for labels in os.listdir(path):

        if labels == 'buildings':

            label = 0

        elif labels == 'forest':

            label = 1

        elif labels == 'glacier':

            label = 2

        elif labels == 'mountain':

            label == 3

        elif labels == 'sea':

            label = 4

        elif labels == 'street':

            label == 5

        

        # For each file name, read the image with openCV and resize it 

        for image_file in os.listdir(path+labels):

            image = cv2.imread(path+labels+r'/'+image_file) #Reading the image (OpenCV)

            image = cv2.resize(image,(150,150)) 

            Images.append(image)

            Labels.append(label)

            

    # Shuffle the data and return it.

    return shuffle(Images,Labels,random_state=817328462)



# Utlity function

def get_classlabel(class_code):

    labels = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

    return labels[class_code]



#Extract the training images from the folders. (This may take a while)

Images, Labels = loadData('../input/seg_train/seg_train/')



Images = np.array(Images)

Labels = np.array(Labels)
print("Shape of Images:",Images.shape)

print("Shape of Labels:",Labels.shape)
f,ax = plot.subplots(5,5) 

f.subplots_adjust(0,0,3,3)

for i in range(0,5,1):

    for j in range(0,5,1):

        rnd_number = randint(0,len(Images))

        ax[i,j].imshow(Images[rnd_number])

        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))

        ax[i,j].axis('off')
# We are going to use the pre-constructed model Inception ResNet v2 

model = InceptionResNetV2(include_top=True, weights=None, input_tensor=None, 

                    input_shape=(150,150,3), pooling=None, classes=6)



model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['acc'])



# The summary is too big too comfort

#model.summary()
trained = model.fit(Images,Labels,epochs=20,batch_size=128, validation_split=0.30)
plot.plot(trained.history['acc'])

plot.plot(trained.history['val_acc'])

plot.title('Model accuracy')

plot.ylabel('Accuracy')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()



plot.plot(trained.history['loss'])

plot.plot(trained.history['val_loss'])

plot.title('Model loss')

plot.ylabel('Loss')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()
test_images,test_labels = loadData('../input/seg_test/seg_test/')

test_images = np.array(test_images)

test_labels = np.array(test_labels)

loss, acc = model.evaluate(test_images,test_labels, verbose=1)

print('Model Test loss: ', loss)

print('Model Test accuracy: ', acc*100, '%')