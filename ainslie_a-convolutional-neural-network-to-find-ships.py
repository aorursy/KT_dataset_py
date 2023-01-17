import numpy as np
import matplotlib.pyplot as plt

# Libraries to handle the data
import json
from PIL import Image # PIL = Python Image Library

file = open('../input/shipsnet.json')
dataset = json.load(file)
file.close()

# Images are 80x80 pixels - Planet data has 3m pixels! Woah! Way finer than the 250m pixels from MODIS!
pixel_width = 80
pixel_height = 80
numChannels = 3 # its 3D because it's RGB image data
input_shape = (pixel_width, pixel_height,numChannels) 

images = []
for index in range( len( dataset['data'] )):
    pixel_vals = dataset['data'][index]
    arr = np.array(pixel_vals).astype('uint8')
    im = arr.reshape((3, 6400)).T.reshape( input_shape )
    images.append( im )
    
images = np.array( images )
labels = dataset['labels']
       
# Inspect an image to make sure it's sensible-looking    
im = Image.fromarray(im)
im.save('80x80.png')

plt.imshow(im)
plt.show()

# Scale by 255.0 to convert RGB pixel values to floats.
images = images / 255.0

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.20)

# We'll build the CNN as a sequence of layers.
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

batch_size = 32 
epochs = 20 #number of times to pass over the training data to fit

# After making a first pass at the CNN, we'll come back and set this Flag
# to True and see if it helps with our accuracy.  
ADD_EXTRA_LAYERS = True

# Initialize the CNN
model = Sequential()

# For the first Convolutional Layer we'll choose 32 filters ("feature detectors"), 
# each with kernel size=(3,3), use activation=ReLU to add nonlinearity
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))

# Downsample by taking Max over (2,2) non-overlapping blocks => helps with spatial/distortion invariance
# with the added benefit of reducing compute time :-)
model.add(MaxPooling2D(pool_size=(2,2)))

# Later we can add extra convolutional layers to see whether they improve the accuracy.
if( ADD_EXTRA_LAYERS ):
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) # Add Dropout layer to reduce overfitting
    
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten all the pooled feature maps into a single vector
model.add(Flatten())

# Append an ANN on the end of the CNN
# Choose 128 nodes on the hidden layer - typically we choose a number that is 
# - not too small so the model can perform well, 
# - and not too large as otherwise the compute time is too long
model.add(Dense(units=128, activation='relu'))

# Final Output Layer has only 1 node. Use activation=sigmoid function as we have a binary outcome (ship/not-ship)
model.add(Dense(units=1, activation='sigmoid'))
# if we had a categorical outcome, we'd use:
#model.add(Dense(units=numberOfCategories, activation='softmax'))

# Compile model - 
# Choose the 'Adam' optimizer for Stochastic Gradient Descent
# https://arxiv.org/pdf/1609.04747.pdf
# For the Loss function we choose binary_crossentropy loss 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

# (if we had a categorical outcome we'd use 'categorical_cross_entropy')
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
# (and then we'd also need to do expand the labels to look like multiple-output nodes! 
# So if the category was 3, then the 3rd node would have a 1 in it!
#y_train = keras.utils.to_categorical( y_train, numberOfCategories)
#y_test = keras.utils.to_categorical( y_test, numberOfCategories )

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
    width_shift_range=0.1,
    height_shift_range=0.1,
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
model.save('findships_model.h5')

# To load our saved model:
from keras.models import load_model
model = load_model('findships_model.h5')

#To make a single prediction (the model expects a four-dimensional input)
test_image = np.expand_dims( images[10], axis = 0)
y_pred = model.predict( test_image )