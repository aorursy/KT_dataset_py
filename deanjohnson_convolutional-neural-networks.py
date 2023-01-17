# Load the libraries
import numpy as np # linear algebra
from sklearn.datasets import load_files # used to enumerate files and output data structure for data
import matplotlib.pyplot as plt # common plot library matplotlib
from PIL import Image # image library PIL will be used to convert images to np arrays

# This is a bit of gto make matplotlib figures appear inline
# in the notebook rather than in a new window
%matplotlib inline
plt.rcParams['figure.figsize'] = (20.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

#kaggle data set location for this dataset
train_dir = '/kaggle/input/fruits/fruits-360/Training/'
test_dir = '/kaggle/input/fruits/fruits-360/Test/'

# this function loads all files and returns 3 objects with the filenames of the images
# you can use the samples parameter to limit the number of samples for quicker testing or if you run out of memory
def load_dataset(path, samples):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files[:samples],targets[:samples],target_labels[:samples]

x_train, y_train,target_labels = load_dataset(train_dir, 20000)
x_test, y_test,_ = load_dataset(test_dir, 5000)

print('Training set size : ' , x_train.shape[0])
print('Testing set size : ', x_test.shape[0])
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#function to translate our image file paths to numpy arrays of image data
def make_image_array(path, image_height, image_width, channels):
    img = load_img(path)  # this is a PIL image
    # Convert to Numpy Array
    x = img_to_array(img)  
    return x.reshape((image_height, image_width, channels))
#load all images as arrays
image_width = 100
image_height = 100
channels = 3
#predefine our array space with the correct shape for efficiency
trainImages = np.ndarray(shape=(len(x_train), image_height, image_width, channels),
                     dtype=np.float32)

#loop through and call our function
i = 0
for _trainFile in x_train:
    trainImages[i] = make_image_array(_trainFile, image_height, image_width, channels)
    i += 1
    
testImages = np.ndarray(shape=(len(x_test), image_height, image_width, channels), 
                        dtype=np.float32)
                        
i = 0
for _testFile in x_test:
    testImages[i] = make_image_array(_testFile, image_height, image_width, channels)
    i += 1

# The following defines a function to view a random sample of the images with class label column headers
def visualize_sample(X_train, y_train, classes, samples_per_class=5):
  num_classes = len(classes)
  for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y) # get all the indexes of class
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs): # plot the image one by one
      plt_idx = i * num_classes + y + 1 # i*num_classes and y+1 determine the row and column respectively
      plt.subplot(samples_per_class, num_classes, plt_idx)
      plt.imshow(trainImages[idx].astype('uint8'))
      plt.axis('off')
      if i == 0:
        plt.title(cls)
  plt.show()
#Get all unique target labels (constitute the classes)
classes = np.unique(target_labels)
print(classes.shape[0])
#Call the function to view the images and classes
visualize_sample(x_train, y_train, classes[: 5], 5)
# Prepare the train and test data for a CNN
# One hot encoding is a way to encode non ordinal categorical data
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(y_train)
one_hot_test_labels = to_categorical(y_test)
# We can see that each category is now represented as a single binary 1 in an array of binary values (the number of classes in our dataset)
print(one_hot_train_labels[0].shape[0])
print(one_hot_train_labels[0])
# Split the data into a train and validation block
from sklearn.model_selection import train_test_split
x_train, y_val, x_lab, y_lab = train_test_split(trainImages, one_hot_train_labels, test_size = 0.2, random_state = 1)
#import tenserflow.keras
from tensorflow.keras import models #used for sequential models
from tensorflow.keras import layers 
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, AveragePooling2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.models import Model #used for functional api
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks

#initialize the sequential model
model = models.Sequential()
#add the hidden layers to the network
#add the first input layer (requires an input shape)
model.add(Conv2D(32, (2, 2), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (4, 4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# a flatten layer is required to transform the 2 dimensional image to a single dimension to input into a dense layer
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# output layer should have a node for each class in this classification problem
model.add(Dense(120))
model.add(Activation('softmax'))

model.compile(optimizer= 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#an early stopper can be added to stop the model training when the model begins to overfit the data:
earlyStopper = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

#the model fit will return the results of each training epoch
history = model.fit(x_train,
                    x_lab,
                    epochs=20,
                    verbose = 1,
                    batch_size=300,
                    validation_data=(y_val, y_lab),
                    callbacks=[earlyStopper])

# In order to visualize the results of training we graph the accuracy on the training set as well as the hold out validation set
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Next we evaluate the model on the yet unseen test data
results = model.evaluate(testImages, one_hot_test_labels)
print(model.metrics_names)
print(results)
# Build and print (plot) the model architecture
visible = Input(shape = (100, 100, 3))
trunk_1 = Conv2D(100, kernel_size=2, activation='relu', strides=2, padding='same')(visible)
trunk_1 = MaxPooling2D(strides=2, padding='same')(trunk_1)
trunk_1 = SeparableConv2D(50, kernel_size=2, strides=1, activation='relu', padding='same')(trunk_1) 
trunk_1 = BatchNormalization()(trunk_1)
branch_1 = Conv2D(50, kernel_size=2, activation='relu', strides=1, padding='same')(trunk_1)
branch_1 = Conv2D(75, kernel_size=4, activation='relu', strides=1)(branch_1)
branch_1 = Conv2D(100, kernel_size=6, activation='relu', strides=2)(branch_1)
branch_2 = Conv2D(50, kernel_size=3, activation='relu', strides=1)(trunk_1)
branch_2 = Conv2D(100, kernel_size=6, activation='relu', strides=2)(branch_2)
trunk_2 = Concatenate(axis=-1)([branch_1, branch_2])
trunk_2 = Conv2D(100, kernel_size=2, activation='relu', strides=1, padding='same')(trunk_2)
trunk_2 = Flatten()(trunk_2)
trunk_2 = Dense(512)(trunk_2)
trunk_2 = Dropout(0.5)(trunk_2)
trunk_2 = Dense(120, activation='softmax')(trunk_2)

model = Model(inputs=visible, outputs=trunk_2)

print(model.summary())
# plot graph
plot_model(model)
#Run the Functional API model (similar to the sequential in how it is setup)
model.compile(optimizer= 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

earlyStopper = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

history = model.fit(x_train,
                    x_lab,
                    epochs=35,
                    verbose = 1,
                    batch_size=500,
                    validation_data=(y_val, y_lab),
                    callbacks=[earlyStopper])

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

results = model.evaluate(testImages, one_hot_test_labels)
print(model.metrics_names)
print(results)