# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import cv2
img = plt.imread('/kaggle/input/face-mask-detection-data/without_mask/Faceimg690.jpg')

plt.imshow(img)
img0 = plt.imread('/kaggle/input/face-mask-detection-data/with_mask/image1,000.jpg')

plt.imshow(img0)
from keras.preprocessing.image import img_to_array,load_img

from keras.applications.mobilenet_v2 import preprocess_input

from keras.utils import to_categorical



X= []

y = []



maskedList = list(os.listdir('../input/face-mask-detection-data/with_mask/'))

unmaskedList = list(os.listdir('../input/face-mask-detection-data/without_mask/'))



for i in maskedList:

    photo = load_img('../input/face-mask-detection-data/with_mask/'+i , target_size=(224,224))

    photo = img_to_array(photo)

    X.append(preprocess_input(photo))

    y.append(1)

    

for i in unmaskedList:

    photo = load_img('../input/face-mask-detection-data/without_mask/'+i,target_size=(224,224))

    photo = img_to_array(photo)

    X.append(preprocess_input(photo))

    y.append(0)

    

X = np.asarray(X)

y = to_categorical(y)
from sklearn.model_selection import train_test_split



Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.30,random_state=32)
from keras.preprocessing.image import ImageDataGenerator



aug = aug = ImageDataGenerator(

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest")
import keras

import tensorflow as tf

from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Input,Dropout
basemodel = keras.applications.MobileNetV2(weights = 'imagenet', include_top=False,input_tensor=Input(shape=(224, 224, 3)))
baseOutput = basemodel.output

headModel = MaxPool2D((7,7))(baseOutput)

headModel = Flatten()(headModel)

headModel = Dense(128,activation='relu')(headModel)

headModel = Dropout(.5)(headModel)

headModel = Dense(2,activation='softmax')(headModel)

model = keras.Model(inputs = basemodel.input,outputs = headModel)



for layer in basemodel.layers:

    layer.trainable = False
model.summary()
from keras.optimizers import Adam



initLR = 1e-4

epochs = 20

batchSize = 32



class myCallback(keras.callbacks.Callback): 

    def on_epoch_end(self, epoch, logs=None): 

        if(logs.get('val_accuracy') > 0.98):   

            print("\nReached %2.2f%% val_accuracy, so stopping training!!" %(.98*100))   

            self.model.stop_training = True

            

opt = Adam(lr=initLR, decay =initLR/epochs)



model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])



history = model.fit(aug.flow(Xtrain,ytrain,batch_size=batchSize),

                    steps_per_epoch=len(Xtrain)//batchSize,

                    validation_data=(Xtest,ytest),

                    validation_steps=len(Xtest)//batchSize,

                    epochs=epochs,

                    callbacks=[myCallback()])
plt.style.use('ggplot')

plt.plot(np.arange(1,8),history.history['loss'],label='loss')

plt.plot(np.arange(1,8),history.history['val_loss'],label='val_loss')

plt.xlabel('Epochs')

plt.ylabel('loss')

plt.legend(loc="lower left")
plt.style.use('ggplot')

plt.plot(np.arange(1,8),history.history['accuracy'],label='accuracy')

plt.plot(np.arange(1,8),history.history['val_accuracy'],label='val_accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc="lower left")
from sklearn.metrics import classification_report

predIdxs = model.predict(Xtest, batch_size=batchSize)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(ytest.argmax(axis=1), predIdxs))
model.save('masked.h5')