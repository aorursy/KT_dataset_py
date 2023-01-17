# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("../input/cifar10Train",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = None )
test_set = test_datagen.flow_from_directory("../input/cifar10Test",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = None)
classifier.fit_generator(training_set,
                         steps_per_epoch = 80,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 20)

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
train_path = r'C:\Users\Sazid\Desktop\thesis\DeepLearningDemos\DeepLearningDemos\cifar10Train'
test_path = r'C:\Users\Sazid\Desktop\thesis\DeepLearningDemos\DeepLearningDemos\cifar10Test'
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),
classes=['dog', 'cat','bird','deer','frog','horse','truck','automobile','airplane','ship'], batch_size=1)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),
classes=['dog', 'cat','bird','deer','frog','horse','truck','automobile','airplane','ship'], batch_size=1)
# plots images with labels within jupyter notebook
def plots(ims, figsize=(4,2), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
imgs, labels = next(train_batches)
plots(imgs, titles=labels)
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()
model.fit_generator(train_batches, steps_per_epoch=4, 
                    validation_data=test_batches, validation_steps=4, epochs=4, verbose=2)
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)
imgs.shape
labels.shape
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
predictions