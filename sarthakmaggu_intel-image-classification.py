import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Activations

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

import keras

from keras.preprocessing.image import ImageDataGenerator 

import os

import matplotlib.pyplot as plt

import cv2

import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from IPython.display import SVG

import seaborn as sns

def get_images(directory):

    Images = []

    Labels = []

    

    for labels in os.listdir(directory):

        if labels == 'glacier': 

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

        

        for image_file in os.listdir(directory+'/'+labels): 

            image = cv2.imread(directory+ '/'+labels+'/'+image_file) 

            image = cv2.resize(image,(150,150)) 

            Images.append(image)

            Labels.append(label)

    

    return shuffle(Images,Labels,random_state=1000) 

def get_category(x):

    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    return labels[x]
Images,Labels = get_images('../input/intel-image-classification/seg_train/seg_train')

Images = np.array(Images)

Labels = np.array(Labels)
category = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

_,count = np.unique(Labels, return_counts = True)

pd.DataFrame({"train": count}, index = category).plot.bar()

plt.show()

plt.pie(count,explode=(0, 0, 0, 0, 0, 0),labels = category,autopct='%1.1f%%')

plt.axis('equal')

plt.title("Propotion")

plt.show()
def display_image(image,label):

    fig = plt.figure(figsize = (10,10))

    fig.suptitle('15 Images from the Dataset', fontsize = 20)

    for i in range(15):

        index = np.random.randint(Images.shape[0])

        plt.subplot(5,5,i+1)

        plt.imshow(image[index])

        plt.xticks([]) #Scale doesn't appear

        plt.yticks([]) #Scale doesn't apper

        plt.title(get_category(label[index]))

        plt.grid(False)

    plt.show()

 #Maximum 25 images can only be displayed.   
display_image(Images, Labels)
print(Images.shape)

print(Labels.shape)
model = Models.Sequential()



model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(3,3))

model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(3,3))

model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(Layers.Conv2D(64,kernel_size=(3,3),activation='relu'))

model.add(Layers.MaxPool2D(3,3))

model.add(Layers.Flatten())

model.add(Layers.Dense(180,activation='relu'))

model.add(Layers.Dense(128,activation='relu'))

model.add(Layers.Dense(64,activation='relu'))

model.add(Layers.Dropout(rate=0.5))

model.add(Layers.Dense(6,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()







trained = model.fit(Images,Labels,epochs=30,validation_split=0.30)
plt.plot(trained.history['accuracy'])

plt.plot(trained.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(trained.history['loss'])

plt.plot(trained.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
train_generator = ImageDataGenerator(rescale = 1/255, zoom_range = 0.3, horizontal_flip = True, rotation_range = 30)

train_generator = train_generator.flow(Images, Labels, batch_size = 64, shuffle = False)
history = model.fit_generator(train_generator, epochs = 30, shuffle = False)
plt.plot(history.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train'], loc='upper left')

plt.show()
test_Images,test_Labels = get_images('../input/intel-image-classification/seg_test/seg_test')

test_Images = np.array(test_Images)

test_Labels = np.array(test_Labels)
test_generator = ImageDataGenerator(rescale = 1/255)

test_generator = test_generator.flow(test_Images, test_Labels, batch_size = 64, shuffle = False)
evaluate = model.evaluate(test_Images, test_Labels, verbose = 1)
print( "Accuracy: "  + str(evaluate[1] * 100) + "%")
evaluate2 = model.evaluate_generator(test_generator, verbose = 1)
print("Accuracy:" + str(evaluate2[1] * 100) + "%")
def get_pred(directory):

    Images = []

    Labels = []

    label = 0

    

    for image_file in os.listdir(directory): 

        image = cv2.imread(directory+ '/' +image_file) 

        image = cv2.resize(image,(150,150)) 

        Images.append(image)

        Labels.append(label)

    

    return shuffle(Images,Labels,random_state=1000) 
pred_Images,pred_Labels = get_pred("../input/intel-image-classification/seg_pred/seg_pred")

pred_Images = np.array(pred_Images)
print(pred_Images.shape)
pred_generator = ImageDataGenerator(rescale = 1/255)

pred_generator = pred_generator.flow(pred_Images, batch_size = 64, shuffle = False)
prediction = model.predict_generator(pred_generator, verbose=1)
prediction.shape