#DINESHSAGAR

#Let's Start, I have built this model in easiest way.

#Any questions or modifications needed? Please ping me. lets solve together :)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Importing Libraries

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
#First lets train the model with train data and after that we will test with test data.

train_folder= '../input/chest-xray-pneumonia/chest_xray/train/'

val_folder = '../input/chest-xray-pneumonia/chest_xray/val/'

test_folder = '../input/chest-xray-pneumonia/chest_xray/test/'

#I haven't seen any image till now, so just lets check how the images are? I chose a random one

plt.imshow(mpimg.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1440-0001.jpeg'))
#Most important step. which is to rescale the images

datagen = ImageDataGenerator(rescale=1. / 255)
datagen=ImageDataGenerator(rescale=1./255)





traingen = datagen.flow_from_directory(train_folder,

                                                 target_size = (200, 200),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



valgen = datagen.flow_from_directory(val_folder,target_size=(200, 200),

                                            batch_size=32,

                                            class_mode='binary')

                                            



testgen = datagen.flow_from_directory(test_folder,

                                            target_size = (200, 200),

                                            batch_size = 32,

                                            class_mode = 'binary')
#The model

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Activation,Dense,Flatten,Dropout



model=Sequential()

model.add(Conv2D(64,kernel_size=(3,3),input_shape=(200,200,3)))

model.add(Activation('relu'))

model.add(MaxPool2D(2,2))



model.add(Conv2D(128,kernel_size=(3,3)))

model.add(Activation('relu'))

model.add(MaxPool2D(2,2))



model.add(Conv2D(128,kernel_size=(3,3)))

model.add(Activation('relu'))

model.add(MaxPool2D(2,2))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dense(64))

model.add(Activation('relu'))



model.add(Dropout(0.2))

model.add(Dense(1))

model.add(Activation('sigmoid'))
from keras.optimizers import Adam



adam=Adam(lr=0.01)



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn_model=model.fit(traingen,epochs=10,validation_data=testgen,steps_per_epoch=10)
model1=model.evaluate(testgen,steps=200)

print(model1)
#Here we are passing 20 images of test folder to check whether it is predicting correctly or not.



x,y = testgen.next()

for i in range(0,20):

  image = x[i]

  label = y[i]

  if label == 0.0:

    print("The X-Ray shown below is suspect of Viral Pneumonia")

  elif label == 1.0:

    print("The X-Ray shown below is suspect of Bacterial Pneumonia")

    print (label)





  plt.imshow(image)

  plt.show()