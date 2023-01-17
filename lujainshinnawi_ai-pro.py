# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image

from google.colab import drive
drive.mount("/drive")


classifier = Sequential()  # Initialise the CNN

#Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# adding a convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

#pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#  Flattening
classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))
#give the binary output
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,validation_split=0.2)

training_set = train_datagen.flow_from_directory('/drive/My Drive/AI pro/dataset',target_size = (64, 64),batch_size = 32,class_mode = 'binary')

testing_set = test_datagen.flow_from_directory('/drive/My Drive/AI pro/dataset',target_size = (64, 64),subset="validation",batch_size = 32,class_mode = 'binary')


classifier.fit_generator(training_set,
steps_per_epoch = training_set.samples/32,epochs = 25,validation_data = testing_set,validation_steps = testing_set.samples/32)
classifier.save("/drive/My Drive/AI pro/module.hd5")

#  Making new predictions

test_image = image.load_img('/drive/My Drive/AI pro/prediction/explosive.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'danger'
else:
    prediction = 'no danger'
