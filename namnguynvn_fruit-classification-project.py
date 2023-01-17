# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import glob
import os
print(os.listdir("../input"))
img=glob.glob(("../input/fruits/fruits-360/Training/*"))
for i in img:
    print(i)


training_fruit_img = []
training_label = []
for dir_path in glob.glob("../input/fruits/fruits-360/Training/*"):
    img_label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        training_fruit_img.append(image)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
len(np.unique(training_label))
label_to_id = {v:k for k,v in enumerate(np.unique(training_label)) }
id_to_label = {v:k for k,v in label_to_id.items() }
id_to_label
print(len(training_label))
training_label_id = np.array([label_to_id[i] for i in training_label])
training_label_id
training_fruit_img.shape,training_label_id.shape
plt.imshow(training_fruit_img[1])
training_label[1]
validation_fruit_img=[]
validation_label =[]
for dir_path in glob.glob("../input/fruits/fruits-360/Test/*"):
    img_label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        validation_fruit_img.append(image)
        validation_label.append(img_label)
validation_fruit_img = np.array(validation_fruit_img)
validation_label = np.array(validation_label)
len(np.unique(validation_label))
validation_label_id = np.array([label_to_id[i] for i in validation_label])
validation_fruit_img.shape,validation_label_id.shape
X_train,X_test = training_fruit_img,validation_fruit_img
Y_train,Y_test =training_label_id,validation_label_id
X_train = X_train/255
X_test = X_test/255

X_flat_train = X_train.reshape(X_train.shape[0],64*64*3)
X_flat_test = X_test.reshape(X_test.shape[0],64*64*3)

#One Hot Encode the Output
Y_train = keras.utils.to_categorical(Y_train, 131)
Y_test = keras.utils.to_categorical(Y_test, 131)

print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print('Flattened:', X_flat_train.shape, X_flat_test.shape)
print(X_train[1200].shape)
plt.imshow(X_train[1200])
plt.show()
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Activation,BatchNormalization
from keras.optimizers import Adamax
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
model0 = Sequential()
model0.add(Conv2D(16,(5,5),input_shape=(64,64,3),padding='same'))
model0.add(LeakyReLU(0.5))
model0.add(BatchNormalization())
model0.add(MaxPooling2D(pool_size=(2,2)))

model0.add(Flatten())
model0.add(Dense(32,activation='relu'))
model0.add(Dropout(0.5))
model0.add(Dense(131))
model0.add(Activation("softmax"))
model0.summary()
model0.compile(loss='categorical_crossentropy',
             optimizer = Adamax(),
             metrics=['accuracy'])

model0.fit(X_train,
          Y_train,
          batch_size=32,
          epochs=5,
          verbose=1,
          validation_data = (X_test,Y_test)
         )
score = model0.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model0.compile(loss='categorical_crossentropy',
             optimizer = Adamax(),
             metrics=['accuracy'])

model0.fit(X_train,
          Y_train,
          batch_size=32,
          epochs=12,
          verbose=1,
          validation_data = (X_test,Y_test)
         )
score = model0.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model = Sequential()
model.add(Conv2D(16,(5,5),input_shape=(64,64,3),padding='same'))
model.add(LeakyReLU(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(5,5),padding='same'))
model.add(LeakyReLU(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(5,5),padding='same'))
model.add(LeakyReLU(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(5,5),padding='same'))
model.add(LeakyReLU(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
#model.add(LeakyReLU(0.1))
model.add(Dropout(0.5))
model.add(Dense(131))
model.add(Activation("softmax"))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer = Adamax(),
             metrics=['accuracy'])

model.fit(X_train,
          Y_train,
          batch_size=32,
          epochs=12,
          verbose=1,
          validation_data = (X_test,Y_test)
         )
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
predictions = model.predict(X_test)
plt.figure(figsize = (64, 64))
it=0
for i in range(1000,20000, 1000):
    plt.subplot(5,4, it + 1)
    it+=1
    plt.xlabel("{}".format(id_to_label[np.argmax(predictions[i])]), fontsize=52)
    plt.imshow(X_test[i])
np.argmax(predictions[2000])
Y_pred=np.zeros(22688)
for i in range(22688):
    Y_pred[i]=np.argmax(predictions[i])
print(Y_pred)
Y_pred.shape

Y_test=validation_label_id
Y_test.shape
id_to_label[np.argmax(predictions[1])]
from sklearn import metrics
cnn_report=metrics.classification_report(Y_test,Y_pred)
print(cnn_report)