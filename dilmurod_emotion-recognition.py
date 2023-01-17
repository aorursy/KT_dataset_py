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

print(os.listdir('../input/data/data/train/'))
#print(os.listdir('../input/data/data/test/surprised/'))

def get_images(directory):

    Images = []

    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street

    label = 0

    

    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.

        if labels == 'angry': #Folder contain Glacier Images get the '2' class label.

            label = 2

        elif labels == 'surprised':

            label = 4

        elif labels == 'disgusted':

            label = 0

        elif labels == 'neutral':

            label = 1

        elif labels == 'fearful':

            label == 5

        elif labels == 'sad':

            label == 3

        elif labels == 'happy':

            label == 6



        

        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder

            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)

            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)

            Images.append(image)

            Labels.append(label)

    

    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepared.



def get_classlabel(class_code):

    labels = {2:'angry', 4:'surprised', 0:'disgusted', 1:'neutral', 5:'fearful', 3:'sad',6:'happy'}

    

    return labels[class_code]

Images, Labels = get_images('../input/data/data/train/') #Extract the training images from the folders.



Images = np.array(Images) #converting the list of images to numpy array.

Labels = np.array(Labels)
Images, Labels = get_images('../input/data/data/train/') #Extract the training images from the folders.



Images = np.array(Images) #converting the list of images to numpy array.

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

SVG(model_to_dot(model).create(prog='dot', format='svg'))

Utils.plot_model(model,to_file='model.png',show_shapes=True)
trained = model.fit(Images,Labels,epochs=35,validation_split=0.30)
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



test_images,test_labels = get_images('../input/data/data/test/')

test_images = np.array(test_images)

test_labels = np.array(test_labels)

model.evaluate(test_images,test_labels, verbose=1)