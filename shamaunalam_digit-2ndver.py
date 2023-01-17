# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data =  pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

target_data , labels =  raw_data.iloc[:,1:],raw_data.iloc[:,0]

print(target_data.head())

print(labels.head())
raw_data.head()
import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense,Flatten,Dense,Dropout,BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import RMSprop

from keras.utils.np_utils import to_categorical

import seaborn as sns

import random

import cv2
n_classes = 10

n_cols = 5

fig,axes = plt.subplots(nrows=n_classes,ncols=n_cols,figsize=(5,10))

fig.tight_layout()



for i in range(n_cols):

    for j in range(n_classes):

        selected_images = raw_data.iloc[:,1:][raw_data.label==j]

        img = np.array(selected_images.iloc[random.randint(0,selected_images.shape[0]),:]).reshape(28,28)

        axes[j][i].imshow(img,cmap='gray')

        axes[j][i].axis('off')



    
sns.countplot(labels)

plt.show()
def load_image_label_array(images_dataframe,label_dataframe):

    x_train = []

    for i in range(images_dataframe.shape[0]):

        image = np.array(images_dataframe.iloc[i,:]).reshape(28,28,1)

        x_train.append(image)

    x_train = np.array(x_train)

    x_train = x_train/255

    y_train = np.array(labels)

    y_train = to_categorical(y_train,10)

   

    return x_train,y_train
x_train,y_train = load_image_label_array(target_data,labels)

print(x_train.shape)

print(y_train.shape)
def create_model():

    model = Sequential()



    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                     activation ='relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))





    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))





    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation = "softmax"))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

model = create_model()

print(model.summary())
h = model.fit(x_train,y_train,verbose=1,epochs=20,batch_size=50,validation_split=0.1)
plt.plot(h.history['loss'],label='loss')

plt.plot(h.history['val_loss'],label='val_loss')

plt.legend()

plt.show()
plt.plot(h.history['accuracy'],label='acc')

plt.plot(h.history['val_accuracy'],label='val_acc')

plt.legend()

plt.show()
model.save('mnist.h5')
xts = []

for i in range(test_data.shape[0]):

    img = np.array(test_data.iloc[i,:]).reshape(28,28)

    xts.append(img)

xts = np.array(xts)
plt.imshow(xts[0])

plt.show()
predictions = []

imageid = []



for i in range(xts.shape[0]):

    image = xts[i]

    image = image.reshape(1,28,28,1)

    image = image/255

    pred = model.predict_classes(image)

    predictions.append(pred[0])

    imageid.append(i+1)

dic = {'ImageId':imageid,'Label':predictions}
pd.DataFrame(dic).to_csv('submission.csv',index=False)