# Check that we have access to the models:
!ls ../input/keras-pretrained-models
# Create paths for model
import os
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Copy model over
!cp ../input/keras-pretrained-models/xception* ~/.keras/models/
# Check that model is in place
!ls ~/.keras/models
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
# Create destination directories
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('train/cat'):
    os.mkdir('train/cat')
if not os.path.exists('train/dog'):  
    os.mkdir('train/dog')
if not os.path.exists('validation'):    
    os.mkdir('validation')
if not os.path.exists('validation/cat'):
    os.mkdir('validation/cat')
if not os.path.exists('validation/dog'):
    os.mkdir('validation/dog')
# define paths
source_path = '../input/dogs-vs-cats-redux-kernels-edition/train/'

cat_train_path = 'train/cat'
dog_train_path = 'train/dog'

cat_validation_path = 'validation/cat'
dog_validation_path = 'validation/dog'
# Loop over image numbering
for i in range(110):
    cat = 'cat.' + str(i) + '.jpg'
    dog = 'dog.' + str(i) + '.jpg'
    # Get source paths
    cat_source = os.path.join(source_path,cat)
    dog_source = os.path.join(source_path,dog)
    # Get destination paths
    if i < 100:
        cat_dest = os.path.join(cat_train_path,cat)
        dog_dest = os.path.join(dog_train_path,dog)
    else: 
        cat_dest = os.path.join(cat_validation_path,cat)
        dog_dest = os.path.join(dog_validation_path,dog)
    # Move file
    copyfile(cat_source,cat_dest)
    copyfile(dog_source,dog_dest)
    print('Copied',(i+1)*2,'out of 220 files',end='\r')
# Check that images are in position
img=mpimg.imread('train/cat/cat.1.jpg')
imgplot = plt.imshow(img)
plt.show()
from keras.applications import Xception
from keras.models import Sequential
model = Xception(weights='imagenet', include_top=False)
model.summary()
from keras.preprocessing.image import ImageDataGenerator

# Only rescaling for training too this time
datagen = ImageDataGenerator(rescale=1./255)
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)
# Set up batch size
batch_size = 1
train_generator = datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size, # How many images do we need at a time
        class_mode=None, # Generator will yield data without labels
        shuffle= False) # Generator will read files in order
validation_generator = validation_datagen.flow_from_directory(
        'validation',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size, # How many images do we need at a time
        class_mode=None,
        shuffle=False)
bottleneck_features_train = model.predict_generator(train_generator, 200, verbose = 1)
import numpy as np
train_labels = np.array([0] * 100 + [1] * 100)
bottleneck_features_validation = model.predict_generator(validation_generator, 20)
validation_labels = np.array([0] * 10 + [1] * 10)
from keras.layers import Flatten, Dense, Dropout, Activation
model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(bottleneck_features_train, train_labels,
          epochs=10,
          batch_size=32,
          validation_data=(bottleneck_features_validation, validation_labels))
# Cleanup for kaggle
!rm -r train
!rm -r validation
