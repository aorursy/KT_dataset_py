import os

import numpy as np

import cv2

import sys

sys.executable

dir = '../input/flowers-recognition/flowers'

a = []

for i in os.listdir(dir):

    print(i)

    a.append(os.path.join(dir,i))
# getting daisy labels + images

data = []

labels = []

for ikl in os.listdir('../input/flowers-recognition/flowers/daisy'):

#         print(ikl)

        path = os.path.join('../input/flowers-recognition/flowers/daisy',ikl)

        img_ray=cv2.imread(path)

#         print(img_ray)

        img_ray = cv2.resize(img_ray,(155,155))

        img_ray = cv2.cvtColor(img_ray,cv2.COLOR_BGR2RGB)

        data.append(img_ray)

        labels.append(0)

# getting rose labels + images

for ilk in os.listdir('../input/flowers-recognition/flowers/rose'):

#         print(ikl)

        path = os.path.join('../input/flowers-recognition/flowers/rose',ilk)

        img_ray=cv2.imread(path)

#         print(img_ray)

        img_ray = cv2.resize(img_ray,(155,155))

        img_ray = cv2.cvtColor(img_ray,cv2.COLOR_BGR2RGB)

        data.append(img_ray)

        labels.append(2)

# getting sunflower labels + images

for ilk in os.listdir('../input/flowers-recognition/flowers/sunflower'):

#         print(ikl)

        path = os.path.join('../input/flowers-recognition/flowers/sunflower',ilk)

        img_ray=cv2.imread(path)

#         print(img_ray)

        img_ray = cv2.resize(img_ray,(155,155))

        img_ray = cv2.cvtColor(img_ray,cv2.COLOR_BGR2RGB)

        data.append(img_ray)

        labels.append(3)
for ilk in os.listdir('../input/flowers-recognition/flowers/tulip'):

#         print(ikl)

        path = os.path.join('../input/flowers-recognition/flowers/tulip',ilk)

        img_ray=cv2.imread(path)

#         print(img_ray)

        img_ray = cv2.resize(img_ray,(155,155))

        img_ray = cv2.cvtColor(img_ray,cv2.COLOR_BGR2RGB)

        data.append(img_ray)

        labels.append(4)
# # getting dandelion labels + images

# for ilk in os.listdir('../input/flowers-recognition/flowers/dandelion'):

# #         print(ikl)

#         path = os.path.join('../input/flowers-recognition/flowers/dandelion',ilk)

# #         print(path)

#         img_ray=cv2.imread(path)

# #         print(img_ray)

#         img_ray = cv2.resize(img_ray,(224,224))

#         img_ray = cv2.cvtColor(img_ray,cv2.COLOR_BGR2RGB)

#         data.append(img_ray)

#         labels.append(1)
import glob

image_list = []

for filename in glob.glob('../input/flowers-recognition/flowers/dandelion/*.jpg'): #assuming gif

    img_ray=cv2.imread(filename)



    img_ray = cv2.resize(img_ray,(155,155))

    img_ray = cv2.cvtColor(img_ray,cv2.COLOR_BGR2RGB)

    data.append(img_ray)

    labels.append(1)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix



from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils import to_categorical



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPool2D, BatchNormalization

 
# import random as rn

# fig,ax=plt.subplots(5,3)

# fig.set_size_inches(15,20)

# for i in range(5):

#     for j in range (3):

#         l=rn.randint(0,len(labels))

#         ax[i,j].imshow(data[l])

#         ax[i,j].set_title('Flower: '+str(labels[l]))


le=LabelEncoder()

labels_all=le.fit_transform(labels)

labels_all=to_categorical(labels_all,5)



data_all = np.array(data)



data_all = data_all/255.0
x_train,x_test,y_train,y_test=train_test_split(data_all,labels_all,test_size=0.20,random_state=42)
## reducing ram

data = None

labels=None

data_all=None

labels_all = None

model = Sequential()

# 1st Convolutional Layer

model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (155,155,3)))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))

# 2nd Convolutional Layer

model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))

# 3rd Convolutional Layer

model.add(Conv2D(filters=156, kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))

# 4th Convolutional Layer

model.add(Conv2D(filters=256,kernel_size = (3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))

# 5th Convolutional Layer

model.add(Conv2D(filters=512,kernel_size = (3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Flatten())

# 1st Fully Connected Layer

model.add(Dense(1024,activation="relu"))

model.add(Dropout(0.5))

model.add(BatchNormalization())

# Add output layer

model.add(Dense(5,activation="softmax"))



model.summary() # print summary my model

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) #compile model
History = model.fit(x_train,y_train,

                              epochs = 200, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // 200)
pred=model.predict(x_test)

pred_digits = np.argmax(pred,axis=1)
count=0

fig,ax=plt.subplots(3,3)

fig.set_size_inches(15,15)

for i in range (3):

    for j in range (3):

        ax[i,j].imshow(x_test[i+j+2])

        ax[i,j].set_title("Predicted Flower : "+str(pred_digits[i+j+2])+"\n Actual :"+ str(np.argmax(y_test[i+j])))

        plt.tight_layout()

#  (le.inverse_transform(np.argmax([y_test[mis_class[count]]]))

        count+=1
model.save('./model200.h5')