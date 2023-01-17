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





import os

mainDIR = os.listdir('../input')

print(mainDIR)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',

                                                    target_size=(32, 32),

                                                    batch_size=8,

                                                    class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',

                                            target_size=(32, 32),

                                            batch_size=8,

                                            class_mode='binary')
from keras.layers import Dense

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.models import Sequential
classifier = Sequential()

classifier.add(Convolution2D(6,(5,5), input_shape=(32,32,3),strides=1, padding='valid', activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

classifier.add(Convolution2D(16,(5,5),strides=1, padding='valid',  activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

classifier.add(Convolution2D(120,(5,5),strides=1, padding='valid',  activation='relu'))

classifier.summary()
classifier.add(Flatten())

classifier.add(Dense(84, input_shape=(120,)))

#classifier.add(Dense(output_dim=84,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.summary()
classifier.fit_generator(training_set,

                            steps_per_epoch=800,

                            epochs=8,

                            validation_data=test_set,

                            validation_steps=200)
import numpy as np

from keras.preprocessing import image

test_image = image.load_img('../input/singleprediction/chestpneumonia.jpg',target_size=(32,32))

test_image=np.expand_dims(test_image,axis=0) #extra dimension is of batch size

#test_image=image.img_to_array(test_image)





print(classifier.predict(test_image))



training_set.class_indices


import os

mainDIR = os.listdir('../kaggle/working')

print(mainDIR)
training_set.class_indices
import pickle

filename = 'finalized_model.sav'

pickle.dump(classifier, open(filename, 'wb'))

 
import numpy as np

from keras.preprocessing import image

test_image = image.load_img('../input/singleprediction/chestpneumonia.jpg',target_size=(32,32))

test_image=np.expand_dims(test_image,axis=0) #extra dimension is of batch size

#test_image=image.img_to_array(test_image)











loaded_model = pickle.load(open(filename, 'rb'))

loaded_model.predict(test_image)