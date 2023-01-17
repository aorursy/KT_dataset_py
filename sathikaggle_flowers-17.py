# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(16, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(8, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Flattening

classifier.add(Flatten())



# Full connection

#classifier.add(Dense(units = 1024, activation = 'relu'))



classifier.add(Dense(units = 512, activation = 'relu'))



classifier.add(Dense(units = 128, activation = 'relu'))



classifier.add(Dense(units = 17, activation = 'softmax'))

# Compiling the CNN

from keras.optimizers import RMSprop, SGD

classifier.compile(loss = 'categorical_crossentropy',

              optimizer = RMSprop(lr = 0.001),

              metrics = ['accuracy'])



#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/flowers/flowers/Train',

                                                 target_size = (128, 128),

                                                 batch_size = 16,

                                                 class_mode = 'categorical',

                                                 shuffle=True)



test_set = test_datagen.flow_from_directory('../input/flowers/flowers/Test',

                                            target_size = (128, 128),

                                            batch_size = 16,

                                            class_mode = 'categorical',

                                            shuffle=False)

nb_train_samples=1071

nb_validation_samples=272

batch_size=16



classifier.fit_generator(training_set,

                         steps_per_epoch = nb_train_samples // batch_size,

                         epochs = 100,

                         validation_data = test_set,

                         validation_steps = nb_validation_samples // batch_size)





#from keras.models import load_model

#classifier.save('flowers_model_train.h5') # creates a HDF5 file ‘my_model.h5’

#del model # deletes the existing model

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('../input/flowers/flowers/Train',

                                                 target_size = (128, 128),

                                                 batch_size = 8,

                                                 class_mode = 'categorical')

import numpy as np

from keras.preprocessing import image

test_image = image.load_img('../input/flowers/flowers/Validation/image14.jpg', 

                            target_size = (128, 128))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = (classifier.predict(test_image))

print(training_set.class_indices)



def predict(a):

    if result[0][0] == 1:

        prediction = 'Type 1'

    elif result[0][1] == 1:

        prediction = 'Type 10'

    elif result[0][2] == 1:

        prediction = 'Type 11'

    elif result[0][3] == 1:

        prediction = 'Type 12'

    elif result[0][4] == 1:

        prediction = 'Type 13'

    elif result[0][5] == 1:

       prediction = 'Type 14'

    elif result[0][6] == 1:

        prediction = 'Type 15'

    elif result[0][7] == 1:

        prediction = 'Type 16'

    elif result[0][8] == 1:

        prediction = 'Type 17'                           

    elif result[0][9] == 1:

        prediction = 'Type 2'

    elif result[0][10] == 1:

        prediction = 'Type 3'

    elif result[0][11] == 1:

        prediction = 'Type 4'

    elif result[0][12] == 1:

        prediction = 'Type 5'

    elif result[0][13] == 1:

        prediction = 'Type 6'

    elif result[0][14] == 1:

        prediction = 'Type 7'

    elif result[0][15] == 1:

        prediction = 'Type 8'

    elif result[0][16] == 1:

        prediction = 'Type 9'

    return prediction 



print("-------------------------------")

print("The test image is {}" .format(predict("result")))

print("-------------------------------")