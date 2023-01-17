# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')

#define file paths.

import os

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten, BatchNormalization

from keras.layers import Dense, Dropout



from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils

import keras

from keras.optimizers import RMSprop



daisy_path = "../input/flower/flower_classification/train/daisy/"

dandelion_path = "../input/flower/flower_classification/train/dandelion/"

rose_path = "../input/flower/flower_classification/train/rose/"

sunflower_path = "../input/flower/flower_classification/train/sunflower/"

tulip_path = "../input/flower/flower_classification/train/tulip/"

test_path="../input/flower/flower_classification/test/"

submission = pd.read_csv('../input/submission.csv')

submission2 = pd.read_csv('../input/flower/flower_classification/submission.csv')
from os import listdir

import cv2







img_data = []

labels = []



size = 64,64

def iter_images(images,directory,size,label):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + images[i])

            img = cv2.resize(img,size)

            img_data.append(img)

            labels.append(label)

    except:

        pass



iter_images(listdir(daisy_path),daisy_path,size,0)

iter_images(listdir(dandelion_path),dandelion_path,size,1)

iter_images(listdir(rose_path),rose_path,size,2)

iter_images(listdir(sunflower_path),sunflower_path,size,3)

iter_images(listdir(tulip_path),tulip_path,size,4)
len(img_data),len(labels)
test_data = []



size = 64,64

def test_images(images,directory,size):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + submission['id'][i]+".jpg")

            img = cv2.resize(img,size)

            test_data.append(img)

    except:

        pass





test_images(listdir(test_path),test_path,size)
len(test_data)
import numpy as np

data = np.asarray(img_data)

testData=np.asarray(test_data)



#div by 255

# data = data / 255.0

testData=testData/255.0



labels = np.asarray(labels)
data.shape,labels.shape
dict = {0:'daisy', 1:'dandelion', 2:'rose', 3:'sunflower', 4:'tulip'}

def plot_image(number):

    fig = plt.figure(figsize = (15,8))

    plt.imshow(testData[number])

    plt.title(dict[labels[number]])
plot_image(0)

labels[0]
from sklearn.model_selection import train_test_split



# Split the data

X_train, X_validation, Y_train, Y_validation = train_test_split(data, labels, test_size=0.02, shuffle= True)
print("Length of X_train:", len(X_train), "Length of Y_train:", len(Y_train))

print("Length of X_validation:",len(X_validation), "Length of Y_validation:", len(Y_validation))
Y_train_one_hot = np_utils.to_categorical(Y_train, 5)

Y_validation_one_hot = np_utils.to_categorical(Y_validation, 5)
classifier = Sequential()



# Convolution layer 1

classifier.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu', border_mode='same', bias=True))



# Pooling layer 1

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.2))



# Convolution and pooling layer 2

classifier.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', bias=True))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.3))



# Classifier and pooling layer 3.

classifier.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', bias=True))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.3))



# Classifier and pooling layer 4.

classifier.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', bias=True))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.3))



# Classifier and pooling layer 5.

classifier.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', bias=True))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.3))



# # Classifier and pooling layer 6.

# classifier.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', bias=True))

# classifier.add(MaxPooling2D(pool_size=(2,2)))

# classifier.add(Dropout(0.3))



# # Classifier and pooling layer 7.

# classifier.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', bias=True))

# classifier.add(Dropout(0.3))





# Flattening

classifier.add(Flatten())



# Full connection

classifier.add(BatchNormalization())

classifier.add(Dense(output_dim = 256, activation='relu'))

classifier.add(Dense(output_dim = 5, activation='softmax'))
# train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True,validation_split=0.1)

train_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.3, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        validation_split=0.1,

        vertical_flip=True)  # randomly flip images

train_set = train_datagen.flow(X_train, Y_train_one_hot, batch_size=32)



# validation_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)

validation_datagen = ImageDataGenerator(rescale=1./255, featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.3, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        validation_split=0.1,

        vertical_flip=True)  # randomly flip images

validation_set = validation_datagen.flow(X_validation, Y_validation_one_hot, batch_size=32)
classifier.summary()
# classifier.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



History =classifier.fit_generator(train_set,

                    steps_per_epoch=3440,epochs=10,

                    validation_data=(validation_set), validation_steps=len(Y_validation), shuffle=True)
# test the accuraccy of validation dataset.

X_validation = (X_validation)*1./255 

scores = classifier.evaluate(X_validation, Y_validation_one_hot, batch_size=32)

print('\nTest result accuracy: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
flower_labels = ['daisy', 'dandelion' ,'rose', 'sunflower', 'tulip']

# def show_test(number):

#     fig = plt.figure(figsize = (15,8))

#     test_image = np.expand_dims(testData[number], axis=0)

#     test_result = classifier.predict_classes(test_image)

#     plt.imshow(testData[number])

#     dict_key = test_result[0]

#     plt.title("Predicted: {}, Actual: {}".format(flower_labels[dict_key], flower_labels[Y_validation[number]]))
# show_test(7)

# show_test(110)

# show_test(156)
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
# getting predictions on val set.

pred_digits=classifier.predict_classes(X_validation)
# now storing some properly as well as misclassified indexes'.

i=0

prop_class=[]

mis_class=[]



for i in range(len(Y_validation)):

    if(Y_validation[i]==pred_digits[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(Y_validation)):

    if(Y_validation[i]!=pred_digits[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(X_validation[prop_class[count]])

        ax[i,j].set_title("Predicted Flower : "+flower_labels[pred_digits[prop_class[count]]]+"\n"+"Actual Flower : "+flower_labels[Y_validation[prop_class[count]]])

        plt.tight_layout()

        count+=1
count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(X_validation[mis_class[count]])

        ax[i,j].set_title("Predicted Flower : "+flower_labels[pred_digits[mis_class[count]]]+"\n"+"Actual Flower : "+flower_labels[Y_validation[mis_class[count]]])

        plt.tight_layout()

        count+=1
pred =  classifier.predict_classes(testData)

newsSbmission=submission

newsSbmission["class"]=pred

newsSbmission.to_csv("submission.csv", index=False)
pred