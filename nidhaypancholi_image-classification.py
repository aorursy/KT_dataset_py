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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

import matplotlib.pyplot as plt
train_datagen=ImageDataGenerator(rescale=1/255,zoom_range=0.2,rotation_range=15,brightness_range=[0.5,1.5],horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1/255)
train=train_datagen.flow_from_directory(

        directory='/kaggle/input/alien-vs-predator-images/alien_vs_predator_thumbnails/data/train',

        class_mode='binary',

        color_mode='rgb',

        target_size=(256,256),

        batch_size=32

)
val=val_datagen.flow_from_directory(

        directory='/kaggle/input/alien-vs-predator-images/alien_vs_predator_thumbnails/data/validation',

        class_mode='binary',

        color_mode='rgb',

        target_size=(256,256),

        batch_size=10

)
model=Sequential()

model.add(Conv2D(32,3,input_shape=(256,256,3)))

model.add(MaxPool2D(2))

model.add(Conv2D(32,3))

model.add(MaxPool2D(3))

model.add(Conv2D(32,3))

model.add(MaxPool2D(2))

model.add(Conv2D(32,3))

model.add(MaxPool2D(2))

model.add(Conv2D(32,3))

model.add(MaxPool2D(2))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.summary()
epoch=20

model.compile(optimizer='rmsprop',metrics='accuracy',loss='binary_crossentropy')

hist=model.fit_generator(train,

                        steps_per_epoch=21,

                        validation_data=val,

                        validation_steps=20,

                        epochs=epoch)
d=hist.history

plt.plot(range(1,epoch+1),d['loss'],label='training loss')

plt.plot(range(1,epoch+1),d['val_loss'],label='validation loss')

plt.legend()
plt.plot(range(1,epoch+1),d['accuracy'],label='training loss')

plt.plot(range(1,epoch+1),d['val_accuracy'],label='validation loss')

plt.legend()
from tensorflow.keras import models
layer_outputs=[layer.output for layer in model.layers[:7]]

model_pred=models.Model(inputs=model.input,outputs=layer_outputs)
from tensorflow.keras.preprocessing.image import load_img,img_to_array

loc='/kaggle/input/alien-vs-predator-images/data/validation/alien/30.jpg'

img=load_img(loc,target_size=(256,256))

img=img_to_array(img)

img_tensor=np.expand_dims(img,axis=0)
preds=model_pred.predict(img_tensor)
first_layer=preds[0]

plt.imshow(first_layer[0,:,:,31])

plt.show()

plt.imshow(preds[1][0,:,:,31])
model.predict(img_tensor)