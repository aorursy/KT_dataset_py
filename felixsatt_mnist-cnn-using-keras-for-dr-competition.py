# Simple CNN for MNIST dataset based on Keras 
# introductary example by their 'Getting Started Guide'
# enhanced with some examples by Tensorflow Documentation and Keras Documentation

from __future__ import print_function

import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization

# basic fitting attributes
batch_size = 64
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

print("#CNN based on Keras for MNIST")
print("#Running for " + str(epochs) + " epochs with " + str(batch_size) + " images batch size")
print("#Image dimensions: " + str(img_rows) + "x " + str(img_cols) + " pixels \n")

# loading test data from csv-files
print('Loading data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('Data loaded.')

# loading samples and labels to numpy arrays
x_train = (train.iloc[:,1:].values).astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
x_test = test.values.astype('float32')

print('\n Training data shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#reshaping data to 2D array (width, height) for plotting and pre-processing
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)

print("Current train image shape: " + str(x_train.shape))

padding = 0
# example images
for i in range(10, 13):
    padding += 1
    plt.subplot(230 + padding)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.xlabel("Label: " + str(y_train[i]))
plt.title("Image examples", fontdict=None, loc='right', pad=10)
plt.show()
# FOR ILLUSTRATION ONLY
# calculating mean and standard deviation
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0).astype(np.float32)

plt.subplot(131)
plt.imshow(mean, cmap=plt.get_cmap('gray'))
plt.xlabel("Mean")
plt.subplot(132)
plt.imshow(std, cmap=plt.get_cmap('gray'))
plt.xlabel("Standard Deviation")
plt.title("Mean and Standard Deviation of training images", fontdict=None, loc='right', pad=10)
plt.show()
# pre-processing using mean / std.-deviation function
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)
def standardize(x): 
    return (x-mean_px)/std_px

#normalization
x_train /= 255
x_test /= 255

#reshaping data to 3-D vector (width, height, depth)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# print out current shape and set numbers
print('Current train image shape: ', x_train.shape)
print(x_train.shape[0], ' training samples')
print(x_test.shape[0], ' testing samples')

# convert class vectors to binary class matrices (one-hot vector); 
# either 'is digit' or 'is not digit'
y_train = keras.utils.to_categorical(y_train, num_classes)
# Linear Sequential model with two Conv2D layers and pooling layers (max pooling) respectively
# Low pass before 2nd Conv2D layer; Batch normalization probably pointless as MNIST-dataset is too small
# Lambda function to pre-process values (subtract mean and divide by standard deviation)
# Kernels are each 3,3 and filters are 32 and 64
# broad dense function with 1024 neurons
model = Sequential()
#model.add(Lambda(standardize, input_shape=input_shape))
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape=input_shape))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size= 2))
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size= 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

# cross entropy function to minimize the distance between calculated probability 
# and 'ground-truth' probability; the idea is to use the expected distribution 
# of probabilities as weights ?? (re-read);
# optimization using a RMS, accuracy metrics for evaluation
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

seed = 42
# 15% of the audgmented data is split for validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=seed)

# data augmentation using generator; rotation, width / height shifting, zooming, noise
augm_data = image.ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, 
                               zoom_range=[0.8,1.2])

augm_train = augm_data.flow(x_train, y_train, batch_size=batch_size)
augm_valid = augm_data.flow(x_valid, y_valid, batch_size=batch_size)

# fitting using generator with adjustable learning rate, validation split and data augmentation
history = model.fit_generator(generator=augm_train,
                              steps_per_epoch=augm_train.n, 
                              epochs=epochs, verbose=1, 
                              validation_data=augm_valid,
                              validation_steps=augm_valid.n)
      
# print final accuracy
print("Estimated Accuracy: " + str(history.history['val_acc'][epochs-1]))
# plotting accuracy results
epoch_plot = range(1, epochs + 1)
plt.plot(epoch_plot, history.history['val_acc'], label='Test Accuracy', c='red')
plt.plot(epoch_plot, history.history['acc'], label='Training Accuracy', c='green')
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy development')
acc = history.history['val_acc'][epochs-1]
acc = round(acc,4)
acc *= 100
plt.text(3,0.95, 'Accuracy after ' + str(epochs)+ ' epochs: ' + str(acc) + '%')
plt.show()

# plotting loss results
plt.plot(epoch_plot, history.history['val_loss'], label='Test Loss')
plt.plot(epoch_plot, history.history['loss'], label='Training Loss')
plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss development')
plt.show()
# make predictions for x_test
predictions = model.predict_classes(x_test, verbose=1)

# submit predictions to .csv
pred_res=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
pred_res.to_csv("mnist_cnn_sattler.csv", index=False, header=True)
print('Loading data...')
ftrain = pd.read_csv("../input/train.csv")
print('Data loaded.')

# loading 30 samples for fourier transform
f_train = (ftrain.iloc[:30,1:].values).astype('float32')

#reshape training data
f_train = f_train.reshape(30, img_rows, img_cols)

print('Performing 2D FFT...')
# convert x_train from time domain to frequency domain via fft
fourier_train = np.fft.fft2(f_train)
print('Done.')
# presenting fourier transformed images
padding = 0
from matplotlib.colors import LogNorm
plt.figure()
for i in range(10,13):
    padding += 1
    plt.subplot(230 + padding)
    plt.imshow(np.abs(fourier_train[i]), norm=LogNorm(vmin=5))
plt.title('Train image after FFT (frequency domain)', fontdict=None, loc='right', pad=10)
plt.show()

