import cv2

import os,glob

from skimage import io,transform

import matplotlib.pyplot as plt

from tensorflow import keras

import numpy as np

import time

def get_images():

    #filenames = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")

    files = glob.glob("/kaggle/input/butterfly-dataset/leedsbutterfly/images/*.png")

    #files=files[:len(files)-int(0.80*len(files))]

    data = []

    labels=[]

    for f1 in files:

        labels.append(int(f1[len(f1)-11:len(f1)-8])-1)

        img = cv2.imread(f1,0)

        img = np.array(transform.resize(img,(46,46),mode="constant"))

        #io.imshow(img)

        #plt.show()

        data.append(img)

        print("Processing image:",len(labels))

        #time.sleep(1)

    return np.array(data),np.array(labels)

images,labels=get_images()

train_images,train_labels=images[:len(labels)-int(0.10*len(labels))],labels[:len(labels)-int(0.10*len(labels))]

test_images,test_labels=images[len(labels)-int(0.10*len(labels)):len(labels)],labels[len(labels)-int(0.10*len(labels)):len(labels)]

print("Total train images",len(train_labels))

print("Total test images:",len(test_labels))

print("Shape train images",train_images.shape)

print("Shape test images",test_images.shape)

print("Shape train labels",train_labels.shape)

print("Shape test labels",test_labels.shape)



train_images=np.expand_dims(train_images,axis=3)

test_images=np.expand_dims(test_images,axis=3)



print("Shape train images",train_images.shape)

print("Shape test images",test_images.shape)

model=keras.models.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(2, 2), input_shape=(46,46,1), activation='relu', padding='same'))

model.add(keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'))

model.add(keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same'))

#model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dropout(rate=0.4))

model.add(keras.layers.Dense(256,activation="relu"))

model.add(keras.layers.Dense(64,activation="relu"))

model.add(keras.layers.Dense(10,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=10,validation_split=.1)

model.save("/kaggle/working/find_butterfly.h5")



print("Loading model.....")

AI_model=keras.models.load_model("/kaggle/working/find_butterfly.h5")

print("Predicting....")

test_loss,test_acc=AI_model.evaluate(test_images,test_labels)

print("accuracy:",test_acc)

predictions = AI_model.predict(test_images)

for i in range(20):

    print("Real_value:",test_labels[i])

    print("Predicted:",np.argmax(predictions[i]))

    #io.imshow(test_images_cpy[i])

    #plt.show()
