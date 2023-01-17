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
from keras.layers import Input,Lambda,Dense,Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

import numpy as np

from glob import glob

import matplotlib.pyplot as plt
IMAGE_SIZE = [224,224] # Lets us set the image size as 224X224

train_path = "../input/cartoon/data/train"

test_path = "../input/cartoon/data/test"
vgg = VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
for layers in vgg.layers:

    layers.trainable = False

    

folder = glob('../input/cartoon/data/train/*')
x = Flatten()(vgg.output)

prediction = Dense(len(folder),activation='softmax')(x)

model = Model(inputs=vgg.input,outputs=prediction)



model.summary()


model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['categorical_accuracy']

                )
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,

                                   zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(train_path,

                                                target_size=(224,224),

                                                batch_size=32,

                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,

                                                target_size=(224,224),

                                                batch_size=32,

                                                class_mode='categorical')
history = model.fit_generator(training_set,

                              validation_data=test_set,

                              epochs=15,

                              verbose=1,

                              steps_per_epoch=len(training_set)/11,

                              validation_steps=len(test_set)

                             )
# loss

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.legend()

plt.show()



# accuracies

plt.plot(history.history['categorical_accuracy'], label='train acc')

plt.plot(history.history['val_categorical_accuracy'], label='val acc')

plt.legend()

plt.show()

val_datagen = ImageDataGenerator()

val_set = val_datagen.flow_from_directory("../input/cartoon/data/validation",

                                                target_size=(224,224),

                                                batch_size=32,

                                                class_mode='categorical')
bean_path = "../input/cartoon/data/validation/bean/0.jpg"

conan_path = "../input/cartoon/data/validation/conan/0.jpg"

doraemon_path = "../input/cartoon/data/validation/doraemon/0.png"

naruto_path = "../input/cartoon/data/validation/naruto/0.jpg"

shinshan_path = "../input/cartoon/data/validation/shinchan/0.jpg"





from keras.models import load_model

import cv2

import numpy as np





img = cv2.imread(shinshan_path)

plt.imshow(img)

img = cv2.resize(img,(240,240))

img = np.reshape(img,[1,240,240,3])



classes = model.predict(img)



index = np.argmax(classes)



if index==0:

    print("It is Bean")

elif index==1:

    print("It is Conan")

elif index==2:

    print("It is Doraemon")

elif index==3:

    print("It is Naruto")

else:

    print("It is Shinchan")
