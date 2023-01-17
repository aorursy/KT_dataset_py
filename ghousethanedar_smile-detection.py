train_folder = '/kaggle/input/smiledetection/datasets/train_folder/'

test_folder = '/kaggle/input/smiledetection/datasets/test_folder/'
!pip install imutils
import numpy as np

import os

import argparse

import cv2

from imutils import paths

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Layer,Conv2D,Activation,MaxPool2D,Dense,Flatten,Dropout

from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix









def dataextractor(path,height=32,width=32):

    data=[]

    labels = []

    imagepaths = list(paths.list_images(path))

    for imagepath in imagepaths:

        image = cv2.imread(imagepath)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image,(height,width),interpolation=cv2.INTER_AREA)

        image = img_to_array(image)

        label = imagepath.split(os.sep)[-2]

        label = int(label)

        labels.append(label)

        data.append(image)

    return np.array(data,dtype='float')/255.0,np.array(labels)

    # splitting the data into train and test



train_X,train_y =dataextractor(train_folder)

test_X,test_y = dataextractor(test_folder)



# (train_X,test_X,train_y,test_y) = train_test_split(data,labels,test_size=0.2,random_state=123)



height = 32

width = 32

depth =1

classes=2



input_shape = (width,height,depth)





















train_y = train_y.reshape((-1,1))

test_y = test_y.reshape((-1,1))
print(train_X.shape)

print(train_y.shape)

print(test_X.shape)

print(test_y.shape)
sgd = SGD(lr=0.01)

model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()
H = model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=15,batch_size=32)





from sklearn.metrics import classification_report

print("[INFO] evaluating network...")

predictions = model.predict(test_X, batch_size=64)

predicted_val = [int(round(p[0])) for p in predictions]

print("classification report ",classification_report(test_y,predicted_val,target_names=['smiling','not_smiling']))


## ploting the training plot and the validation plot



plt.figure(figsize=(10,8))

plt.plot(np.arange(0,15),H.history['loss'],label='loss')

plt.plot(np.arange(0,15),H.history['val_loss'],label='val_loss')

plt.plot(np.arange(0,15),H.history['accuracy'],label='accuracy')

plt.plot(np.arange(0,15),H.history['val_accuracy'],label='val_accuracy')

plt.xlabel('Epchos')

plt.ylabel('Percentage')

plt.title('Training &validation Loss and accuracy Plot')

plt.legend()

plt.show()

print("[INFO] serializing network...")

model.save('smile.hdf5')