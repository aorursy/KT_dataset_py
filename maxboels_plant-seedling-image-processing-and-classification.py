import cv2

from glob import glob

import numpy as np

from matplotlib import pyplot as plt

import math

import pandas as pd



ScaleTo = 70  # px to scale

seed = 7  # fixing random



path = '../input/plant-seedlings-classification/train/*/*.png' 

files = glob(path) #return a list of pathnames



trainImg = []

trainLabel = []

j = 1

num = len(files)



# Obtain images and resizing, obtain labels

for img in files:

    #print(str(j) + "/" + str(num), end="\r")

    trainImg.append(cv2.resize(cv2.imread(img), (ScaleTo, ScaleTo)))  # Get image (with resizing)

    trainLabel.append(img.split('/')[-2])  # Get image label (folder name)

    j += 1



trainImg = np.asarray(trainImg)  # Train images set

trainLabel = pd.DataFrame(trainLabel)  # Train labels set
# Show 8 example images

for i in range(8):

    plt.subplot(2, 4, i + 1) #2 rows, 4 columns and index = i+ 1

    plt.imshow(trainImg[i])
clearTrainImg = []

examples = []; getEx = True

for img in trainImg:

    # Use gaussian blur

    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   

    

    # Convert to HSV image

    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  

    

    # Create mask (parameters - green color range)

    lower_green = (25, 40, 50)

    upper_green = (75, 255, 255)

    mask = cv2.inRange(hsvImg, lower_green, upper_green)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    # Create bool mask

    bMask = mask > 0  

    

    # Apply the mask

    clear = np.zeros_like(img, np.uint8)  # Create empty image

    clear[bMask] = img[bMask]  # Apply boolean mask to the origin image

    

    clearTrainImg.append(clear)  # Append adds and object to the end of a list (image without backgroung)

    

    # Show examples

    if getEx:

        plt.subplot(2, 3, 1); plt.imshow(img)  # Show the original image

        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Blurred image

        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV image

        plt.subplot(2, 3, 4); plt.imshow(mask)  # Mask

        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Boolean mask

        plt.subplot(2, 3, 6); plt.imshow(clear)  # Image without background

        getEx = False



clearTrainImg = np.asarray(clearTrainImg)
# Show sample result

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(clearTrainImg[i])
clearTrainImg = clearTrainImg / 255
from keras.utils import np_utils

from sklearn import preprocessing

import matplotlib.pyplot as plt



# Encode labels and create classes

le = preprocessing.LabelEncoder()

le.fit(trainLabel[0])

print("Classes: " + str(le.classes_))

encodeTrainLabels = le.transform(trainLabel[0])



# Make labels categorical

clearTrainLabel = np_utils.to_categorical(encodeTrainLabels)

num_clases = clearTrainLabel.shape[1]

print("Number of classes: " + str(num_clases))



# Plot of label types numbers

trainLabel[0].value_counts().plot(kind='bar')
from sklearn.model_selection import train_test_split



trainX, testX, trainY, testY = train_test_split(clearTrainImg, clearTrainLabel, 

                                                test_size=0.1, random_state=seed, 

                                                stratify = clearTrainLabel)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        rotation_range=180,  # randomly rotate images in the range

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally

        height_shift_range=0.1,  # randomly shift images vertically 

        horizontal_flip=True,  # randomly flip images horizontally

        vertical_flip=True  # randomly flip images vertically

    )  

datagen.fit(trainX)
import numpy

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import BatchNormalization



numpy.random.seed(seed)  # Fix seed



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(ScaleTo, ScaleTo, 3), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(num_clases, activation='softmax'))



model.summary()



# compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger



# learning rate reduction

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.4, 

                                            min_lr=0.00001)



# checkpoints

filepath="drive/DataScience/PlantReco/weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 

                             verbose=1, save_best_only=True, mode='max')

filepath="drive/DataScience/PlantReco/weights.last_auto4.hdf5"

checkpoint_all = ModelCheckpoint(filepath, monitor='val_acc', 

                                 verbose=1, save_best_only=False, mode='max')



# all callbacks

callbacks_list = [checkpoint, learning_rate_reduction] # removed checkpoint_all from []



# fit model

#hist = model.fit_generator(datagen.flow(trainX, trainY, batch_size=75), 

#                           epochs=35, validation_data=(testX, testY), 

#                          steps_per_epoch=trainX.shape[0], callbacks=callbacks_list)
model.load_weights("../input/plantrecomodels/weights.best_17-0.96.hdf5")



data = np.load("../input/plantrecomodels/Data.npz")

d = dict(zip(("trainX","testX","trainY", "testY"), (data[k] for k in data)))

trainX = d['trainX']

testX = d['testX']

trainY = d['trainY']

testY = d['testY']



print(model.evaluate(trainX, trainY))  # Evaluate on train set

print(model.evaluate(testX, testY))  # Evaluate on test set
from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    fig = plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

predY = model.predict(testX)

predYClasses = np.argmax(predY, axis = 1) 

trueY = np.argmax(testY, axis = 1) 



# confusion matrix

confusionMTX = confusion_matrix(trueY, predYClasses) 



# plot the confusion matrix

plot_confusion_matrix(confusionMTX, classes = le.classes_) 
path = '../input/plant-seedlings-classification/test/*.png'

files = glob(path)



testImg = []

testId = []

j = 1

num = len(files)



# Obtain images and resizing, obtain labels

for img in files:

    print("Obtain images: " + str(j) + "/" + str(num), end='\r')

    testId.append(img.split('/')[-1])  # Images id's

    testImg.append(cv2.resize(cv2.imread(img), (ScaleTo, ScaleTo)))

    j += 1



testImg = np.asarray(testImg)  # Train images set



for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(testImg[i])
clearTestImg = []

examples = []; getEx = True

for img in testImg:

    # Use gaussian blur

    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   

    

    # Convert to HSV image

    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  

    

    # Create mask (parameters - green color range)

    lower_green = (25, 40, 50)

    upper_green = (75, 255, 255)

    mask = cv2.inRange(hsvImg, lower_green, upper_green)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    # Create bool mask

    bMask = mask > 0  

    

    # Apply the mask

    clear = np.zeros_like(img, np.uint8)  # Create empty image

    clear[bMask] = img[bMask]  # Apply boolean mask to the origin image

    

    clearTestImg.append(clear)  # Append image without backgroung

    

    # Show examples

    if getEx:

        plt.subplot(2, 3, 1); plt.imshow(img)  # Show the original image

        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Blur image

        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV image

        plt.subplot(2, 3, 4); plt.imshow(mask)  # Mask

        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Boolean mask

        plt.subplot(2, 3, 6); plt.imshow(clear)  # Image without background

        getEx = False



clearTestImg = np.asarray(clearTestImg)
# Show sample result

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(clearTestImg[i])
clearTestImg = clearTestImg / 255
pred = model.predict(clearTestImg)

# Write result to file

predNum = np.argmax(pred, axis=1)

predStr = le.classes_[predNum]



res = {'file': testId, 'species': predStr}

res = pd.DataFrame(res)

res.to_csv("res.csv", index=False)