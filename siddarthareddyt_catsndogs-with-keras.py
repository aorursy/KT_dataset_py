# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense, Dropout

#print(check_output(["ls", "../input/dataset/dataset/training_set/cats"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
shape_w = 60

shape_h = 60

batchsize = 32
classifier = Sequential()



classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(shape_w, shape_h, 3)))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Conv2D(32, (3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Conv2D(64, (3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation="relu"))

classifier.add(Dropout(0.5))

classifier.add(Dense(units=1, activation="sigmoid"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



validation_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/training_set',

                                                 target_size = (shape_w, shape_h),

                                                 batch_size = batchsize,

                                                 class_mode = 'binary')



validation_set = validation_datagen.flow_from_directory('../input/dataset/dataset/test_set',

                                            target_size = (shape_w, shape_h),

                                            batch_size = batchsize,

                                            class_mode = 'binary')
classifier.fit_generator(training_set,

                         steps_per_epoch = (10000 // batchsize),

                         epochs = 50,

                         validation_data = validation_set,

                         validation_steps = (2500 // batchsize))



classifier.save('catsdogs.h5')

classifier.save_weights('catsdogs_weights.h5')