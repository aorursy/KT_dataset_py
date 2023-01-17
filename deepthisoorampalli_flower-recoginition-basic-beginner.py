# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers-recognition/flowers/flowers"))
# Any results you write to the current directory are saved as output.
#importing keras libraries and packages

from keras.models import Sequential # to initiale the NN
from keras.layers import Conv2D # Add covnet layers , 2D for images
from keras.layers import MaxPooling2D # Add pooling layers
from keras.layers import Flatten # to convert into vector for fully connected layer
from keras.layers import Dense # to add layers

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#initializing CNN
classifier = Sequential()

#Add Convolution layer
classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3),activation = 'relu'))

#Add pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#ADD second convolution layer
classifier.add(Conv2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Add flattening
classifier.add(Flatten())

#Add full connection
classifier.add(Dense(units = 128 , activation = "relu" , kernel_initializer = "uniform"))
#output layer
classifier.add(Dense(units= 5 , activation = "sigmoid" , kernel_initializer = "uniform"))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy' ,metrics = ['accuracy'])
#Model Summary
classifier.summary()
#Part 2
#fitting CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory('../input/flowers-recognition/flowers/flowers',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
classifier.fit_generator(
        training_set,
        steps_per_epoch=4323,
        epochs=5,
        )
#save the trained model
classifier.save('flower_recogintion.h5')
#In order to load the saved model we can use "load_weights" parameter
classifier.load_weights('flower_recogintion.h5')
#make predictions for new image
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('../input/test-image/Sunflower_or_rose.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Daisy'
elif result[0][1]==1:
    prediction = 'Dandelion'
elif result[0][2]==1:
    prediction = 'Rose'
elif result[0][3]==1:
    prediction='Sunflower'
elif result[0][4]==1:
    prediction='Tulip'
print('so the flower is %s' %prediction)