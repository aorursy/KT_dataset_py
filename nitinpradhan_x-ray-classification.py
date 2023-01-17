# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



# Step 2 - Max Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())
# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
import os

paths = os.listdir(path="../input")

print(paths)

path_train = "../input/chest_xray/chest_xray/train"

path_test = "../input/chest_xray/chest_xray/test"

#create datagenetaror to preprocess and create batches of image to have more number of images

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



#read images from directory and resize(as done before during convolution) them for training and testing

training_set = train_datagen.flow_from_directory(path_train,

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory(path_test,

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')
#fit the images 

classifier.fit_generator(training_set,

                         steps_per_epoch = 5216,

                         epochs = 5,

                         validation_data = test_set,

                         validation_steps = 624)