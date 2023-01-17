# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization
from keras.layers import MaxPooling2D,Dropout
from keras.layers import Flatten
from keras.layers import Dense
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array
import random
data=[]
labels=[]
random.seed(42)
imagePaths = sorted(list(os.listdir("../input/natural-images/data/natural_images")))
random.shuffle(imagePaths)

for img in imagePaths:
    path=sorted(list(os.listdir("../input/natural-images/data/natural_images/"+img)))
    for i in path:
        image = cv2.imread("../input/natural-images/data/natural_images/"+img+'/'+i)
        image = cv2.resize(image, (128,128))
        image = img_to_array(image)
        data.append(image)
 
        l = label = img
        labels.append(l)
        
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)
print(labels[0])
(x_train,x_test,y_train,y_test)=train_test_split(data,labels,test_size=0.1,random_state=42)
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu',padding='same'))
# Adding a second convolutional layer
#classifier.add(BatchNormalization(axis=1))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


classifier.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))
# Adding a second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))
#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))
# Adding a second convolutional layer
classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))
#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(Dense(1024,activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

# Step 4 - Full connection
classifier.add(Dense(output_dim = 8, activation = 'softmax'))

# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   shear_range=0.2,

                                  )

#test_datagen = ImageDataGenerator(rescale = 1./255)


#training_set = train_datagen.flow_from_directory('../input/natural-images/data/natural_images',
#                                                target_size = (128,128),
#                                               batch_size = 512,
#                                              class_mode = 'categorical')

#testing_set = train_datagen.flow_from_directory('../input/data/natural_images',
#                                                 target_size = (64,64),
#                                                 batch_size = 8,
#                                                 class_mode = 'categorical')



classifier.fit_generator(train_datagen.flow(x_train,y_train,batch_size=512),
                         epochs = 20,
                         steps_per_epoch=10,
                         validation_data=(x_test,y_test),
                         )

from keras.preprocessing.image import img_to_array
import numpy as np 
list=['airplane','car','cat','dog','flower','fruit','motorbike','person']
image = cv2.imread('../input/natural-images/data/natural_images/fruit/fruit_0076.jpg')
type(image)
 
# pre-process the image for classification
image = cv2.resize(image, (128,128))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


pred = classifier.predict(image)[0]
print((pred))
for i in range(8):
    if pred[i]>0.5:
        print(list[i],(pred[i]).astype('float32'))
    

print(pred)
