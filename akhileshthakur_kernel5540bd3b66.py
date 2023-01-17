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
!unzip /kaggle/input/dogs-vs-cats/train.zip

!unzip /kaggle/input/dogs-vs-cats/test1.zip
import os

all_files = os.listdir('/kaggle/working/train/')

print(all_files[0] , type(all_files[0]))
all_files[0].split('.')
file = []

Animal = []

for i in os.listdir('/kaggle/working/train/'):

    file.append(i)

    animal_name = i.split('.')[0]

    Animal.append(animal_name)
import pandas as pd

import numpy as np

import keras
train_df = pd.DataFrame({'IMAGE_NAME':file , 'CATEGORY':Animal})



train_df.head()
from keras.preprocessing.image import ImageDataGenerator



data_gen = ImageDataGenerator(

    rotation_range=20 ,

    rescale=1/.255 ,

    horizontal_flip=True,

    validation_split=0.2

)
train_generator = data_gen.flow_from_dataframe(

    train_df,

    directory = '/kaggle/working/train/',

    x_col='IMAGE_NAME',

    y_col='CATEGORY',

    target_size = (224,224),

    class_mode='categorical',

    batch_size=32,

    shuffle=True,

    subset='training'

)

validation_generator = data_gen.flow_from_dataframe(

    train_df,

    directory = '/kaggle/working/train/',

    x_col='IMAGE_NAME',

    y_col='CATEGORY',

    target_size = (224,224),

    class_mode='categorical',

    batch_size=32,

    shuffle=True,

    subset='validation'

)
import keras

from keras.layers import Dense, Dropout , Flatten

from keras.layers import Conv2D, MaxPooling2D, Activation , GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers
vgg_model = keras.applications.vgg19.VGG19(include_top=False,input_shape=(224,224,3))



model = vgg_model.layers[-2].output

model = GlobalAveragePooling2D()(model)

model = Dense(1024,activation='relu')(model)

model =(Dropout(0.25)(model))

model = (Dense(512,activation='relu')(model))



model = Dropout(0.25)(model)

model = (Dense(2,activation='softmax')(model))

         

nw_model = keras.Model(inputs = vgg_model.input , outputs=model)
for layer in nw_model.layers:

    layer.trainable = False

for layer in nw_model.layers[-7:-1]:

    layer.trainable=True
nw_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nw_model.fit_generator(train_generator, epochs=15 , 

                   validation_data=validation_generator,)