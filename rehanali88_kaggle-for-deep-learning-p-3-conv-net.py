# the following line will make plots appear with out calling plt.show()
%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # I/O of data sets
from zipfile import ZipFile # woking with zip archives
import os # directory operations

# packages for visualization
import ipywidgets as iw
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.image as im 
from keras.preprocessing import image

# keras for deep learning
from keras import layers, models, optimizers, metrics 
from keras.preprocessing.image import ImageDataGenerator 
zp_filenames = ['Testing.zip','Training.zip','Validation.zip']
# using list comprehension to build a list for zipped file with their path names.
zp_filenamesWithpath = [os.path.join('../input/kaggle-for-deep-learning-p-1-getting-data-ready',k) for k in zp_filenames]
for k in zp_filenamesWithpath: # looping over the files that need to be unzipped
    
    # extracting the zipfiles to the current directory
    with ZipFile(k,'r') as zp: 
        zp.extractall()
# using dict comprehension to store paths to a dictionary
path = {i: os.path.join('working', i) for i in os.listdir('working')} 

path
# Reading the '.csv' file to a dataframe.
CnnP = pd.read_csv('../input/kaggle-for-deep-learning-p-2-visualization-eda/CNNParameters.csv', index_col = 0)
# Converting the data frame to a dictionary and printing it's values.
CnnP = CnnP.to_dict()['0']
print(CnnP)
# Initializing a sequential model
model = models.Sequential()

# Adding convolution and maxpooling layers
model.add(layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        input_shape=(100, 100, 3))) # convolution layer with an output of 32
model.add(layers.MaxPooling2D((2, 2))) # max pooling layer

model.add(layers.Conv2D(64, (3, 3), 
                        activation='relu')) # convolution later
model.add(layers.MaxPooling2D((2, 2))) # maxpooling layer

model.add(layers.Conv2D(128, (3, 3), 
                        activation='relu')) # convolution layer
model.add(layers.MaxPooling2D((2, 2))) # maxpooling layer

model.add(layers.Conv2D(128, (3, 3), 
                        activation='relu')) # convolution layer
model.add(layers.MaxPooling2D((2, 2))) # max pooling layer

# adding a flatten and a dense layer
model.add(layers.Flatten()) # flatten layer and a dropout layer
model.add(layers.Dense(512, 
                       activation='relu')) # dense layer

# output layer with 'softmax' activation as there are multiple classes.
model.add(layers.Dense(CnnP['No classes'], 
                       activation='softmax'))
# Checking the model architecture
model.summary()
model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.Adam(lr=1e-4),
              metrics = [metrics.categorical_accuracy])
# Initializing the ImageDataGenerator object. We set rescale to 1/255 to normalize pixel values to be in range 0,1
datagen = ImageDataGenerator(rescale=1./255)
# Using the generator to get training images ready for the network
train_generator = datagen.flow_from_directory(path['Training'],
                                              target_size = (100,100),
                                              batch_size = CnnP['Training batch size'],
                                              class_mode = 'categorical')
# Using the generator to get validation images ready for the network
validation_generator = datagen.flow_from_directory(path['Validation'],
                                                   target_size = (100,100),
                                                   batch_size = CnnP['Validation batch size'],
                                                   class_mode = 'categorical')
# Fitting the model
history = model.fit_generator(train_generator,
                              steps_per_epoch = CnnP['Training steps/epoch'],
                              epochs = 50,
                              validation_data = validation_generator,
                              validation_steps = CnnP['Validation steps/epoch'])
model_json = model.to_json() # serialize model to JSON
# writing the model as a json file
with open("fruit_classification_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("fruit_classification_weights.h5") # saving model weights as HDF5
print("Model Saved")
def plotModel(history):
    fg, (ax1, ax2) = plt.subplots(2,1,figsize = (15, 15)) # creating subplots
    # calculating percentage accuracy and loss for training and validation
    acc = np.array(history.history['categorical_accuracy'])*100
    val_acc = np.array(history.history['val_categorical_accuracy'])*100
    loss = history.history['loss']
    val_loss = history.history['val_loss']
 
    epochs = range(1, len(acc) + 1) # calculating x-axis values
    fnt = {'fontsize': 20,
           'fontweight' : 'medium',
           'verticalalignment': 'baseline'} # dictionary to control font for labels & title

    # plotting accuracy
    ax1.plot(epochs, acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.set_title('Training and validation accuracy', fnt)
    ax1.set_xlabel('Epoch number', fnt, labelpad = 20)
    ax1.set_ylabel('Accuracy (%)', fnt)
    ax1.legend(fontsize = 20)

    # plotting loss
    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.set_title('Training and validation loss', fnt)
    ax2.set_xlabel('Epoch number', fnt, labelpad = 20)
    ax2.set_ylabel('Loss', fnt)
    ax2.legend(fontsize = 20)

    # changing font size of tick labels
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    return fg
    
fg = plotModel(history)    
fg.savefig('modelResults.png')# saving the figure
# Creating the test_generator
test_generator = datagen.flow_from_directory(path['Test'],
                                             target_size = (100,100),
                                             batch_size = 1,
                                             class_mode = 'categorical') # testing in bathches of 5
# Calculating the test loss and accuracy
test_loss, test_acc = model.evaluate_generator(test_generator, steps = 3155)
# printing the test loss and accuracy
print("Test accuracy for the model is %0.2f %s. \n" % (test_acc*100, '%'))
print('Test loss for the model is %0.2f .' % test_loss)
model2 = models.Sequential()

model2.add(layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        input_shape=(100, 100, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), 
                        activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), 
                        activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), 
                        activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Flatten()) 

model2.add(layers.Dropout(0.5)) # adding a drop out layer

model2.add(layers.Dense(512, 
                       activation='relu'))
model2.add(layers.Dense(CnnP['No classes'], 
                       activation='softmax')) 
# Making an image data generator object with augmentation 
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# This list contains names of images in folder named orange
imgs_orange = [os.path.join(path['Test'],'Orange',img) for img in os.listdir(os.path.join(path['Test'],'Orange'))]
img = image.load_img(imgs_orange[0])
img_ar = image.img_to_array(img) # convert image to a numpy array
img_ar = img_ar.reshape((1,) + img_ar.shape) # reshape the image
# Creating a figure with 4 subplots
fg, ax = plt.subplots(4, 1, figsize = (20,20))

# variable for referring to an axis
k = 0 

# Using train_datagen to generate randomly transformed images
for batch in train_datagen.flow(img_ar, batch_size=1):
    
    ax[k].imshow(image.array_to_img(batch[0]))
    ax[k].axis('off')
    k += 1 # updating value of k
    if k == 4: # we only want to see 4 images (we have 4 subplots so lets break if k is equal to 4)
        break
        
fg.savefig('augmentedImages.png')# saving the figure
datagen = ImageDataGenerator(rescale=1./255) 
# Using the generator for training directory
train_generator = train_datagen.flow_from_directory(path['Training'],
                                              target_size = (100,100),
                                              batch_size = CnnP['Training batch size'],
                                              class_mode='categorical')
# Using the generator validation directory
validation_generator = datagen.flow_from_directory(path['Validation'],
                                                   target_size = (100,100),
                                                   batch_size = CnnP['Validation batch size'],
                                                   class_mode = 'categorical')
model2.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.Adam(lr=1e-4),
              metrics = [metrics.categorical_accuracy])
# Fitting the model
history2 = model2.fit_generator(train_generator,
                              steps_per_epoch = CnnP['Training steps/epoch'],
                              epochs = 50,
                              validation_data = validation_generator,
                              validation_steps = CnnP['Validation steps/epoch'])
model_json = model2.to_json() # serialize model to JSON
# writing the model as a json file
with open("fruit_classification_model_dropout.json", "w") as json_file:
    json_file.write(model_json)
model2.save_weights("fruit_classification_weights_augmented.h5") # saving model weights as HDF5
print("Model Saved")
fg = plotModel(history2)    
fg.savefig('modelResultsAugmentation.png')# saving the figure
# Calculating the test loss and accuracy
test_loss2, test_acc2 = model2.evaluate_generator(test_generator, steps = 3155)
# printing the test loss and accuracy
print("Test accuracy for the model is %0.2f %s. \n" % (test_acc2*100, '%'))
print('Test loss for the model is %0.2f .' % test_loss2)
import shutil
shutil.rmtree('working')
fg, (ax1, ax2) = plt.subplots(2,1,figsize = (15, 15)) # creating subplots

# calculating percentage accuracy and loss for training and validation
acc1 = np.array(history.history['categorical_accuracy'])*100
val_acc1 = np.array(history.history['val_categorical_accuracy'])*100

loss1 = history.history['loss']
val_loss1 = history.history['val_loss']

acc2 = np.array(history2.history['categorical_accuracy'])*100
val_acc2 = np.array(history2.history['val_categorical_accuracy'])*100

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
 
epochs = range(1, len(acc1) + 1) # calculating x-axis values

fnt = {'fontsize': 20,
        'fontweight' : 'medium',
        'verticalalignment': 'baseline'} # dictionary to control font for labels & title

# plotting accuracy
ax1.plot(epochs, acc1, 'bo', label='Training acc')
ax1.plot(epochs, val_acc1, 'b', label='Validation acc')

ax1.plot(epochs, acc2, 'ro', label='Training acc, augmented')
ax1.plot(epochs, val_acc2, 'r', label='Validation acc, augmented')

ax1.set_title('Training and validation accuracy', fnt)
ax1.set_xlabel('Epoch number', fnt, labelpad = 20)
ax1.set_ylabel('Accuracy (%)', fnt)
ax1.legend(fontsize = 20)

# plotting loss
ax2.plot(epochs, loss1, 'bo', label='Training loss')
ax2.plot(epochs, val_loss1, 'b', label='Validation loss')

ax2.plot(epochs, loss2, 'ro', label='Training loss, augmented')
ax2.plot(epochs, val_loss2, 'r', label='Validation loss, augmented')

ax2.set_title('Training and validation loss', fnt)
ax2.set_xlabel('Epoch number', fnt, labelpad = 20)
ax2.set_ylabel('Loss', fnt)
ax2.legend(fontsize = 20)

# changing font size of tick labels
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    