#Importing the keras 

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

#Initialising the CNN

classifier = Sequential()



#Step 1 - Convolution

#32 convolution kernels, size of the kernel is 3 x 3

#bigger size masks will blur the image

#64, 64 image size 3 is color, all images are not 64 X 64 it will resize

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



#Step 2 - Pooling

#Stride is 2 X 2

classifier.add(MaxPooling2D(pool_size = (2, 2)))



#Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))



#Flattening

classifier.add(Flatten())



#Full Connection

#here we don't need to give input layer no.of units it comes from flatten

#this is hidden layer with 128 units (first layer o/p units=128)

#units=1 because it is binary class classification

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))



#Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 

                   metrics = ['accuracy'])





#Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/training_set',

                                                target_size = (64, 64),

                                                batch_size = 32,

                                                class_mode = 'binary')



test_set = test_datagen.flow_from_directory('../input/dataset/dataset/test_set',

                                                target_size = (64, 64),

                                                batch_size = 32,

                                                class_mode = 'binary')



classifier.fit_generator(training_set,

                        steps_per_epoch = 8000//32,

                        epochs = 20,

                        validation_data = test_set,

                        validation_steps = 2000//32)

#Testing with single prediction image

import numpy as np

from keras.preprocessing import image

test_image = image.load_img('../input/dataset/dataset/single_prediction/cat_or_dog_1.jpg',

                           target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)

training_set.class_indices

if result[0][0] == 1:

    prediction = 'dog'

    print("The test image is")

    print(prediction)

else:

    prediction = 'cat'

    print("The test image is")

    print(prediction)
