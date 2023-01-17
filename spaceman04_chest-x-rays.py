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


from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense

from keras.models import Sequential

from keras.layers import GlobalAveragePooling2D

model = InceptionV3(weights = 'imagenet',include_top=False, input_shape=(299, 299, 3))

#model.summary()



#Creating a Sequential type model

classifier = Sequential()





# freeze all layers of the pre-trained model

for layer in model.layers:

    layer.trainable = False

    print(layer)

#classifier.add(Dense(units = 4, activation = 'relu'))

classifier.add(model) #Adding the InceptionV3 layer to our classifier

classifier.add(GlobalAveragePooling2D())

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.summary()    





#Compiling the CNN

classifier.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])









#Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set  = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',

                                                  target_size=(299, 299),

                                                  batch_size=2,

                                                  class_mode='binary')



test_set = test_datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/test',

        target_size=(299, 299),

        batch_size=2,

        class_mode='binary')



classifier.fit_generator(training_set,

                         steps_per_epoch = 800,

                         epochs = 20,

                         validation_data = test_set,

                         validation_steps = 200)
