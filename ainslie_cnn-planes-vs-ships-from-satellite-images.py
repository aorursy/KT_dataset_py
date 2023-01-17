import numpy as np
import matplotlib.pyplot as plt

# We'll build the CNN as a sequence of layers.
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Libraries to handle the data
import json
from PIL import Image # PIL = Python Image Library
from skimage.transform import resize
file = open('../input/planesnet/planesnet.json')
dataset = json.load(file)
file.close()

# Plane Images are 20x20 pixels 
pixel_width = 20
pixel_height = 20
numChannels = 3 # its 3D because it's RGB image data
input_shape = (pixel_width, pixel_height,numChannels) 

images = []
for index in range( len( dataset['data'] )):
    pixel_vals = dataset['data'][index]
    arr = np.array(pixel_vals).astype('uint8')
    arr = arr / 255.0 # Need to scale this here as shipimages will be fractional after downsampling
    im = arr.reshape(( numChannels, pixel_width * pixel_height)).T.reshape( input_shape )
    images.append( im )
           
file = open('../input/ships-in-satellite-imagery/shipsnet.json')
shipdataset = json.load(file)
file.close()
    
# Ship Images are 80x80 pixels 
shippixel_width = 80
shippixel_height = 80
shipnumChannels = 3 # its 3D because it's RGB image data
shipinput_shape = (shippixel_width, shippixel_height,shipnumChannels) 

for index in range( len( shipdataset['data'] )):
    pixel_vals = shipdataset['data'][index]
    arr = np.array(pixel_vals).astype('uint8')
    im = arr.reshape((3, shippixel_width * shippixel_height)).T.reshape( shipinput_shape )
    image_resized = resize(im, (pixel_width, pixel_height),mode='constant')
    # this returns the image_resized in RGB format as (float,float,float)
    images.append( image_resized )
    
images = np.array( images )
labels = np.array(dataset['labels'])

shiplabels = 2* np.array(shipdataset['labels'])
labels = np.hstack( (labels, shiplabels))
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.20)

batch_size = 32 
epochs = 10 #number of times to pass over the training data to fit

# After making a first pass at the CNN, we'll come back and set this Flag
# to True to improve our accuracy by adding extra layers 
ADD_EXTRA_LAYERS = True

# Initialize the CNN as a sequence of layers
model = Sequential()

# For the first Convolutional Layer we'll choose 32 filters ("feature detectors"), 
# each with kernel size=(3,3), use activation=ReLU to add nonlinearity
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))

# Downsample by taking Max over (2,2) non-overlapping blocks => helps with spatial/distortion invariance
# with the added benefit of reducing compute time :-)
model.add(MaxPooling2D(pool_size=(2,2)))

# Later we can add extra convolutional layers to improve our accuracy
if( ADD_EXTRA_LAYERS ):
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) # Add Dropout layer to reduce overfitting
    
    model.add(Conv2D(64, (3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))
    model.add(Dropout(0.5)) # Add Dropout layer to reduce overfitting
    
# Flatten all the pooled feature maps into a single vector
model.add(Flatten())

# Append an ANN on the end of the CNN
# Choose 256 and then 128 nodes on the hidden layers - typically we choose a number that is 
# - not too small so the model can perform well, 
# - and not too large as otherwise the compute time is too long
if( ADD_EXTRA_LAYERS ):
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    
model.add(Dense(units=128, activation='relu'))

numberOfCategories = 3
model.add(Dense(units=numberOfCategories, activation='softmax'))

# Compile model - 
# Choose the 'Adam' optimizer for Stochastic Gradient Descent
# https://arxiv.org/pdf/1609.04747.pdf
# We have a categorical outcome so we use 'categorical_cross_entropy'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# We need to expand the labels to look like multiple-output nodes! 
# So if the category was 3, then the 3rd node would have a 1 in it!
y_train = keras.utils.to_categorical( y_train, numberOfCategories)
y_test = keras.utils.to_categorical( y_test, numberOfCategories )

# We use Image Augmentation as the number of images is small.
# (We generate extra training images by applying various distortions to the samples
# in our training set. This increases the size of our training set and so helps reduce
# overfitting.) 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    # These first four parameters if true would result in scaling of the input images,  
    # which in this situation reduce the ability of the CNN to train properly.
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True)

training_set = train_datagen.flow(x_train, y_train, batch_size=batch_size)

test_datagen = ImageDataGenerator()
test_set = test_datagen.flow(x_test, y_test, batch_size=batch_size)
# fits the model on batches with real-time data augmentation:
model.fit_generator( training_set,
                    steps_per_epoch=len(x_train) / batch_size, 
                    validation_data=test_set,
                    validation_steps=len(x_test)/batch_size,
                    epochs=epochs)
