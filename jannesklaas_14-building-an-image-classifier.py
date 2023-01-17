# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import matplotlibs image tool
import matplotlib.image as mpimg
# Flip the switch to get easier matplotlib rendering
%matplotlib inline
# for file listing
import os
# for file moving
from shutil import copyfile
len(os.listdir('../input/train'))
# Create destination directories
os.mkdir('train')
os.mkdir('train/cat')
os.mkdir('train/dog')
os.mkdir('validation')
os.mkdir('validation/cat')
os.mkdir('validation/dog')
# define paths
source_path = '../input/train'

cat_train_path = 'train/cat'
dog_train_path = 'train/dog'

cat_validation_path = 'validation/cat'
dog_validation_path = 'validation/dog'
# Loop over image numbering
for i in range(12500):
    cat = 'cat.' + str(i) + '.jpg'
    dog = 'dog.' + str(i) + '.jpg'
    # Get source paths
    cat_source = os.path.join(source_path,cat)
    dog_source = os.path.join(source_path,dog)
    # Get destination paths
    if i < 12000:
        cat_dest = os.path.join(cat_train_path,cat)
        dog_dest = os.path.join(dog_train_path,dog)
    else: 
        cat_dest = os.path.join(cat_validation_path,cat)
        dog_dest = os.path.join(dog_validation_path,dog)
    # Move file
    copyfile(cat_source,cat_dest)
    copyfile(dog_source,dog_dest)
    print('Copied',(i+1)*2,'out of 25,000 files',end='\r')
img=mpimg.imread('train/cat/cat.1.jpg')
imgplot = plt.imshow(img)
plt.show()
img=mpimg.imread('train/dog/dog.4.jpg')
imgplot = plt.imshow(img)
plt.show()
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
# The keras backend module has information stored about how Keras wants to interact with its backend (TensorFlow)
from keras import backend as K

# We will resize all images to 150 by 150 pixels
img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
# First conv layer:
# Filters: 32
# Kernel: 3x3 (9px)
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
# Max pool
# Size: 2x2 (combine 4px to 1)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second conv layer
# Filters: 32
# Kernel: 3x3 (9px)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Max pool
# Size: 2x2 (combine 4px to 1)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third conv layer
# Filters: 64
# Kernel: 3x3 (9px)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# Max pool
# Size: 2x2 (combine 4px to 1)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten turns 3D map into vector
model.add(Flatten())
# Dense layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40, # Rotate up to 40 degrees
        width_shift_range=0.2, # Shift width by max 20%
        height_shift_range=0.2, # Shift height by max 20%
        rescale=1./255, # Rescale pixel activations from 0 to 255 (as it is in files) to 0 to 1
        shear_range=0.2, # Cut away max 20% of the image
        zoom_range=0.2, # Zoom in 20% max
        horizontal_flip=True, # Flip image randomly
        fill_mode='nearest') # Fill missing pixels with the nearest value
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)
# Set up batch size
batch_size = 16
train_generator = datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size, # How many images do we need at a time
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        'validation',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size, # How many images do we need at a time
        class_mode='binary')
history = model.fit_generator(
        train_generator, # Get training data from training generator
        steps_per_epoch=12000 // batch_size, # // is the integer rounded division
        epochs=3, # train for 50 epochs
        validation_data=validation_generator, # Validate with validation generator
        validation_steps=1000 // batch_size)
plt.plot(history.history['acc'])
train_generator.class_indices
# Load the whole image module
from keras.preprocessing import image
# Load numpy, how did we live without it so far??
import numpy as np
# Specify the file path
img_path = 'validation/dog/dog.12001.jpg'
# Load and resize the image
img = image.load_img(img_path, target_size=(150, 150))
# Turn the image into a numpy array
x = image.img_to_array(img)
# Resize the matrix to meet the expected shape
x = np.expand_dims(x, axis=0)
# Rescale the image
x = x / 255

# Obtain prediction
features = model.predict(x)
# Output prediction
print('Dog probability: ' + str(features))
img=mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()
wrong_classifications = []
for i in range(12000,12500):
    img_path = 'validation/dog/dog.' +str(i)+ '.jpg'
    # Load and resize the image
    img = image.load_img(img_path, target_size=(150, 150))
    # Turn the image into a numpy array
    x = image.img_to_array(img)
    # Resize the matrix to meet the expected shape
    x = np.expand_dims(x, axis=0)
    # Rescale the image
    x = x / 255

    # Obtain prediction
    features = model.predict(x)
    if features < 0.5:
        wrong_classifications.append(img_path)
        print('Actual: Dog, Predicted: Cat')
        img=mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
    if len(wrong_classifications) > 5:
        break
wrong_classifications = []
for i in range(12000,12500):
    img_path = 'validation/cat/cat.' +str(i)+ '.jpg'
    # Load and resize the image
    img = image.load_img(img_path, target_size=(150, 150))
    # Turn the image into a numpy array
    x = image.img_to_array(img)
    # Resize the matrix to meet the expected shape
    x = np.expand_dims(x, axis=0)
    # Rescale the image
    x = x / 255

    # Obtain prediction
    features = model.predict(x)
    if features > 0.5:
        wrong_classifications.append(img_path)
        print('Actual: Cat, Predicted: Dog')
        img=mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
    if len(wrong_classifications) > 5:
        break
model.save_weights('first_try.h5')
# Cleanup for kaggle
# Kaggle only allows a certain amount of output files so we have to remove 
# our resorted training data at the end of the kernel
!rm -r train
!rm -r validation