import numpy as np 

import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
image_gen = ImageDataGenerator(

                               width_shift_range=0.1, 

                               height_shift_range=0.1, 

                               rescale=1/255, 

                               shear_range=0.2, 

                               zoom_range=0.2, 

                               fill_mode='nearest'

                              )
image_shape = (350,350,1)

batch_size = 16

train_image_gen = image_gen.flow_from_directory('../input/recognizance/train',color_mode='grayscale',

                                               target_size=image_shape[:2],

                                               batch_size=batch_size,

                                               class_mode='binary',seed=42)
from keras.models import Sequential

from keras.layers import Activation, Flatten, Dense, Conv2D, MaxPool2D
model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(350,350,1), activation='relu',))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(filters=128, kernel_size=(3,3),activation='relu',))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(512))

model.add(Activation('relu'))





model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])





model.fit_generator(generator=train_image_gen,epochs=20)
model.fit_generator(generator=train_image_gen,epochs=5)
model.fit_generator(generator=train_image_gen,epochs=5)
import os

files = []



for dirname, _, filenames in os.walk('/kaggle/input/recognizance/test/'):

    for filename in filenames:

        files.append(filename)
pred = []

for i in files:

    img = image.load_img('../input/recognizance/test/'+i, target_size=(350,350,1),color_mode = "grayscale")

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img/255

    pred.append(model.predict_classes(img)[0][0])
sub = pd.DataFrame(pred,files).reset_index()

sub.columns = ['image','label']

sub.to_csv('sub4.csv',index=False)
!gzip sub4.csv
model.save("model.h5")
model.save_weights("modelweights.h5")
from IPython.display import FileLink

FileLink(r'modelweights.h5')