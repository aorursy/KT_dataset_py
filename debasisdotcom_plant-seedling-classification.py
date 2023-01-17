import os

from tqdm.notebook import tqdm_notebook as tqdm

import cv2

import numpy as np

import pandas as pd

from glob import glob

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.utils import np_utils

import tensorflow as tf

import keras

from keras.applications.mobilenet import MobileNet, preprocess_input



from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns



from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
CLASS = {

    'Black-grass': 0,

    'Charlock': 1,

    'Cleavers': 2,

    'Common Chickweed': 3,

    'Common wheat': 4,

    'Fat Hen': 5,

    'Loose Silky-bent': 6,

    'Maize': 7,

    'Scentless Mayweed': 8,

    'Shepherds Purse': 9,

    'Small-flowered Cranesbill': 10,

    'Sugar beet': 11

}



INV_CLASS = {CLASS[j]:j for j in CLASS}
def preprop_img(image_path, verbose=0):

    if verbose:

        print(image_path)

    img=cv2.imread(image_path)

    img=cv2.resize(img, (128,128))

    return img
#Reading the image file and converting them to array

train_image=[]

train_label=[]

BASE='../input/plant-seedlings-classification/train'

for i in tqdm(os.listdir(BASE), total=len(CLASS)):

    for j in os.listdir(os.path.join(BASE,i)):

        train_image.append(preprop_img(os.path.join(BASE,i,j)))

        train_label.append(CLASS[i])

train_image=np.array(train_image)

train_label=np.array(train_label)



print("Shape of train_image:",train_image.shape,"Shape of train_label:",train_label.shape)
train_label_cat = keras.utils.to_categorical(train_label,len(CLASS))

print(train_label_cat.shape)
plt.figure(figsize=(12,12))



for i in range(12):  

    

    plt.subplot(3,4,i+1)

    

    index = np.where(train_label==i)[0][1]

    plt.imshow(train_image[index])

    plt.title(INV_CLASS[np.argmax(train_label_cat[index])])

    plt.xticks([]), plt.yticks([])



plt.suptitle("Visualization of Plant Seedlings", fontsize=20)    

plt.tight_layout()

plt.show()
clearTrainImg = []

examples = []; getEx = True

plt.figure(figsize=(10,9))



for img in train_image:

    

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

    

    clearTrainImg.append(clear)  # Append image without backgroung

    

    # Show examples

    if getEx:

        plt.subplot(2, 3, 1); plt.imshow(img)  # Show the original image

        plt.xticks([]), plt.yticks([]), plt.title("Original Image")

        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Blur image

        plt.xticks([]), plt.yticks([]), plt.title("Blur Image")

        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV image

        plt.xticks([]), plt.yticks([]), plt.title("HSV Image")

        plt.subplot(2, 3, 4); plt.imshow(mask)  # Mask

        plt.xticks([]), plt.yticks([]), plt.title("Mask")

        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Boolean mask

        plt.xticks([]), plt.yticks([]), plt.title("Boolean mask Image")

        plt.subplot(2, 3, 6); plt.imshow(clear)  # Image without background

        plt.xticks([]), plt.yticks([]), plt.title("Image without background")

        getEx = False



plt.suptitle("Masking the Plant", fontsize=20)

plt.tight_layout()
# Visulaising the sample result

clearTrainImg = np.asarray(clearTrainImg)

plt.figure(figsize=(12,8))



for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(clearTrainImg[i])

    plt.xticks([]), plt.yticks([])

    

plt.suptitle("Sample result", fontsize=20)  

plt.tight_layout()

plt.show()
# Plot of label types numbers

classes = list(INV_CLASS.values())



sns.set_style('darkgrid')  

ax = sns.countplot(x=0, data=pd.DataFrame(train_label))

ax.set_xticklabels(classes)



plt.xticks(rotation=90)

plt.show()
clearTrainImg = clearTrainImg / 255
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(clearTrainImg,train_label_cat, shuffle=True, test_size=0.2)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
X_train=X_train.astype('float32') 

X_test=X_test.astype('float32')
datagen = ImageDataGenerator(

        rotation_range=180,  # randomly rotate images in the range

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally

        height_shift_range=0.1,  # randomly shift images vertically 

        horizontal_flip=True,  # randomly flip images horizontally

        vertical_flip=True  # randomly flip images vertically

    )  

datagen.fit(X_train)
#Got 90 % test accuracy on model built from scratch



tf.keras.backend.clear_session() #clear the weights



np.random.seed(2)  # Fix seed



model = Sequential([Conv2D(filters=64, kernel_size=(5, 5), input_shape=(128, 128, 3), activation='relu'),

                    BatchNormalization(axis=3),

                    Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),

                    MaxPooling2D((2, 2)),

                    BatchNormalization(axis=3),

                    Dropout(0.1),

                    

                    Conv2D(filters=128, kernel_size=(5, 5), activation='relu'),

                    BatchNormalization(axis=3),

                    Conv2D(filters=128, kernel_size=(5, 5), activation='relu'),

                    MaxPooling2D((2, 2)),

                    BatchNormalization(axis=3),

                    Dropout(0.1),

                   

                    Conv2D(filters=256, kernel_size=(5, 5), activation='relu'),

                    BatchNormalization(axis=3),

                    Conv2D(filters=128, kernel_size=(5, 5), activation='relu'),

                    MaxPooling2D((2, 2)),

                    BatchNormalization(axis=3),

                    Dropout(0.1),

                   

                    Flatten(),

                    

                    Dense(256, activation='relu'),

                    BatchNormalization(),

                    Dropout(0.5),

                   

                    Dense(256, activation='relu'),

                    BatchNormalization(),

                    Dropout(0.5),

                   

                    Dense(12, activation='softmax')])







model.summary()



# compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# ALPHA = 1.0



# # mnet = MobileNet(input_shape=(128,128,3), include_top=False, alpha=ALPHA)
# # Got 91.26% accuracy on validation dataset on 10 epochs and with batch size as 32.

# tf.keras.backend.clear_session()



# for layers in mnet.layers:

#       layers.trainable = False



# model = Sequential([mnet,

#                     Flatten(),

                    

#                     Dense(256, activation='relu'),

#                     BatchNormalization(),

#                     Dropout(0.5),

                   

#                     Dense(256, activation='relu'),

#                     BatchNormalization(),

#                     Dropout(0.5),

                    

#                     Dense(12,activation='softmax')])



# model.summary()



# # compile model

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])            
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.4, 

                                            min_lr=0.00001)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),epochs=5,

                    validation_data=(X_test, Y_test),

                    steps_per_epoch=(X_train.shape[0]),

                    verbose=1,

                    callbacks=[learning_rate_reduction])
plt.figure(figsize=(20,10))

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
path = '../input/plant-seedlings-classification/test/*.png'

files = glob(path)



testImg = []



for img in files:

    testImg.append(cv2.resize(cv2.imread(img), (128, 128)))



testImg = np.asarray(testImg)  # Train images set



plt.figure(figsize=(10,6))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(testImg[i])

    

plt.suptitle("Visualising the test dataset", fontsize=20)    

plt.tight_layout()

plt.show()
clearTestImg = []

examples = []; getEx = True

plt.figure(figsize=(10,9))



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

        plt.xticks([]), plt.yticks([]), plt.title("Original Image")

        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Blur image

        plt.xticks([]), plt.yticks([]), plt.title("Blur Image")

        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV image

        plt.xticks([]), plt.yticks([]), plt.title("HSV Image")

        plt.subplot(2, 3, 4); plt.imshow(mask)  # Mask

        plt.xticks([]), plt.yticks([]), plt.title("Mask")

        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Boolean mask

        plt.xticks([]), plt.yticks([]), plt.title("Boolean mask Image")

        plt.subplot(2, 3, 6); plt.imshow(clear)  # Image without background

        plt.xticks([]), plt.yticks([]), plt.title("Image without background")

        getEx = False



plt.suptitle("Masked Test Image", fontsize=20)

plt.tight_layout()

clearTestImg = np.asarray(clearTestImg)
#Normalizing the test data

clearTestImg = clearTestImg / 255
pred = model.predict(clearTestImg)

predNum = np.argmax(pred, axis=1)
testId = []

for i in files:

    testId.append(i.split('/')[-1]) 
predStr=[]

for i in predNum:

    predStr.append(INV_CLASS[i])
# # Write result to file

# PS = {'file': testId, 'species': predStr}

# PS = pd.DataFrame(res)

# PS.to_csv("PS.csv", index=False)
plt.figure(figsize=(12,12))



for i,j in enumerate(files[:12]):  

    

    plt.subplot(3,4,i+1)

    

    img = np.array(cv2.imread(j))

    plt.imshow(img)

    plt.title(predStr[i])

    plt.xticks([]), plt.yticks([])



plt.suptitle("Visualization of Predicted Plant Seedlings", fontsize=20)    

plt.tight_layout()

plt.show()