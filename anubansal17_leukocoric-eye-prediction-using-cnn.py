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

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout
classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape =(128, 128, 3), activation = 'relu'))

#Step2-Max Pooling(Size of resulting matrix after applying max pooling is (Size of feature map)/2

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32, (3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step3-Flattening

classifier.add(Flatten())

#Step4-Full Connection

classifier.add(Dense(activation="relu", units=128))

classifier.add(Dense(activation="sigmoid", units=1))

#Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



training_set = train_datagen.flow_from_directory(   '../input/train data/Train data',

                                                    target_size=(128, 128),

                                                    batch_size=32,

                                                    class_mode='binary')



classifier.fit_generator(training_set,

                    steps_per_epoch=885,

                    epochs=20)
import numpy as np

from keras.preprocessing import image

result = []

for i in range(0, 200):

    test_image = image.load_img('../input/evaluation data/Evaluation data/'+str(i)+'.jpg', target_size = (128, 128))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    pred = int(classifier.predict(test_image))

    if pred == 0 :

        pred = 1

    else :

        pred = 0

    result.append(pred) 

result = np.array(result)

print(result)

np.size(result)
import pandas as pd

pd.DataFrame(result).to_csv("../file3.csv",header = 'category', index=1)