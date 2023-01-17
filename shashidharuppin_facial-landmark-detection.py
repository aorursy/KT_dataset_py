# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the dependencies

from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt

%matplotlib inline 
#Reading the Lookup Table

IdLookupTable = pd.read_csv('/kaggle/input/IdLookupTable.csv')

IdLookupTable.info()
IdLookupTable.head()
#Reading the training dataset

training = pd.read_csv('/kaggle/input/training/training.csv')

training.info()
training.head()
test = pd.read_csv('/kaggle/input/test/test.csv')

test.info()
test.head()
#There are lot of null values in the training dataset. Let's drop all the rows which have null values. 

training = training.dropna()
training.info()
training.shape, type(training)
# Let us seperate images and keypoints seperately for the train dataset



#Writing a function to seperate the images from the training data

def load_images(image_data):

    images = []

    for idx, sample in image_data.iterrows():

        image = np.array(sample['Image'].split(' '), dtype=int)

        image = np.reshape(image, (96,96,1))

        images.append(image)

    images = np.array(images)/255.

    return images



#Writing a function to seperate the keypoints from the training data

def load_keypoints(keypoint_data):

    keypoint_data = keypoint_data.drop('Image',axis = 1)

    keypoint_features = []

    for idx, sample_keypoints in keypoint_data.iterrows():

        keypoint_features.append(sample_keypoints)

    keypoint_features = np.array(keypoint_features, dtype = 'float')

    return keypoint_features





clean_train_images =load_images(training)

print("Shape of clean_train_images: {}".format(np.shape(clean_train_images)))



clean_train_keypoints = load_keypoints(training)

print("Shape of clean_train_keypoints: {}".format(np.shape(clean_train_keypoints)))



test_images = load_images(test)

print("Shape of test_images: {}".format(np.shape(test_images)))
#Let's create a function to plot the image

def plot_sample(image, keypoint, axis, title):

    image = image.reshape(96,96)

    axis.imshow(image, cmap='gray')

    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)

    plt.title(title)
#Let's create a new label for images and keypoints

train_images = clean_train_images

train_keypoints = clean_train_keypoints

fig, axis = plt.subplots()

plot_sample(clean_train_images[25], clean_train_keypoints[25], axis, "Sample Image & Keypoints")
#Various Image Agumentation choices

sample = 25

horizontal_flip = True

rotation_augmentation = True

brightness_augmentation = True

shift_augmentation = True

random_noise_augmentation = True



#Function for Horizantal flipping of images

def left_right_flip(images, keypoints):

    flipped_keypoints = []

    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)

    for idx, sample_keypoints in enumerate(keypoints):

        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping

    return flipped_images, flipped_keypoints



if horizontal_flip:

    flipped_train_images, flipped_train_keypoints = left_right_flip(clean_train_images, clean_train_keypoints)

    print("Shape of flipped_train_images:",np.shape(flipped_train_images))

    print("Shape of flipped_train_keypoints:",np.shape(flipped_train_keypoints))

    

    #Concatenating the train images with flipped image & train keypoints with flipped train points

    train_images = np.concatenate((train_images, flipped_train_images))

    train_keypoints = np.concatenate((train_keypoints, flipped_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(flipped_train_images[sample], flipped_train_keypoints[sample], axis, "Horizontally Flipped") 
#Example for converting rotated image

#img = cv2.imread('messi5.jpg',0)

#rows,cols = img.shape



#M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)

#dst = cv2.warpAffine(img,M,(cols,rows))
import cv2

from math import sin, cos, pi





rotation_angles = [12]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)

pixel_shifts = [12]    # Horizontal & vertical shift amount in pixels (includes shift from all 4 corners)



#Writing a function for Rotation of the Images

def rotate_augmentation(images, keypoints):

    rotated_images = []

    rotated_keypoints = []

    print("Augmenting for angles (in degrees): ")

    

    for angle in rotation_angles:    # Rotation augmentation for a list of angle values

        for angle in [angle,-angle]:

            print(f'{angle}', end='  ')

            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)

            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)

            

            # For train_images

            for image in images:

                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)

                rotated_images.append(rotated_image)

            

            # For train_keypoints

            for keypoint in keypoints:

                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension

                for idx in range(0,len(rotated_keypoint),2):

                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point

                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)

                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)

                rotated_keypoint += 48.   # Add the earlier subtracted value

                rotated_keypoints.append(rotated_keypoint)

            

    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints



#For more details on the transformation of the images below is the link.

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html



if rotation_augmentation:

    rotated_train_images, rotated_train_keypoints = rotate_augmentation(clean_train_images, clean_train_keypoints)

    print("\nShape of rotated_train_images:",np.shape(rotated_train_images))

    print("Shape of rotated_train_keypoints:\n",np.shape(rotated_train_keypoints))

    

    #Concatenating the train images with rotated image & train keypoints with rotated train points

    train_images = np.concatenate((train_images, rotated_train_images))

    train_keypoints = np.concatenate((train_keypoints, rotated_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(rotated_train_images[sample], rotated_train_keypoints[sample], axis, "Rotation Augmentation")

#Writing a function for Brightness Alteration

def alter_brightness(images, keypoints):

    altered_brightness_images = []

    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]

    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]

    altered_brightness_images.extend(inc_brightness_images)

    altered_brightness_images.extend(dec_brightness_images)

    return altered_brightness_images, np.concatenate((keypoints, keypoints))



if brightness_augmentation:

    altered_brightness_train_images, altered_brightness_train_keypoints = alter_brightness(clean_train_images, clean_train_keypoints)

    print("Shape of altered_brightness_train_images:",np.shape(altered_brightness_train_images))

    print("Shape of altered_brightness_train_keypoints:",np.shape(altered_brightness_train_keypoints))

    

    #Concatenating the train images with brightness altered images & train keypoints with brightness altered train points

    train_images = np.concatenate((train_images, altered_brightness_train_images))

    train_keypoints = np.concatenate((train_keypoints, altered_brightness_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(altered_brightness_train_images[sample], altered_brightness_train_keypoints[sample], axis, "Increased Brightness") 

    fig, axis = plt.subplots()

    plot_sample(altered_brightness_train_images[len(altered_brightness_train_images)//2+sample], altered_brightness_train_keypoints[len(altered_brightness_train_images)//2+sample], axis, "Decreased Brightness") 
#Writing a function for shift the image horizontal and verical

def shift_images(images, keypoints):

    shifted_images = []

    shifted_keypoints = []

    for shift in pixel_shifts:    # Augmenting over several pixel shift values

        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:

            M = np.float32([[1,0,shift_x],[0,1,shift_y]])

            for image, keypoint in zip(images, keypoints):

                shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)

                shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])

                if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):

                    shifted_images.append(shifted_image.reshape(96,96,1))

                    shifted_keypoints.append(shifted_keypoint)

    shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)

    return shifted_images, shifted_keypoints



if shift_augmentation:

    shifted_train_images, shifted_train_keypoints = shift_images(clean_train_images, clean_train_keypoints)

    print(f"Shape of shifted_train_images:",np.shape(shifted_train_images))

    print(f"Shape of shifted_train_keypoints:",np.shape(shifted_train_keypoints))

    

    train_images = np.concatenate((train_images, shifted_train_images))

    train_keypoints = np.concatenate((train_keypoints, shifted_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(shifted_train_images[sample], shifted_train_keypoints[sample], axis, "Shift Augmentation")
#Writing a function to add noise

def add_noise(images):

    noisy_images = []

    for image in images:

        noisy_image = cv2.add(image, 0.008*np.random.randn(96,96,1))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]

        noisy_images.append(noisy_image.reshape(96,96,1))

    return noisy_images



if random_noise_augmentation:

    noisy_train_images = add_noise(clean_train_images)

    print("Shape of noisy_train_images:",np.shape(noisy_train_images))

    

    train_images = np.concatenate((train_images, noisy_train_images))

    train_keypoints = np.concatenate((train_keypoints, clean_train_keypoints))

    fig, axis = plt.subplots()

    plot_sample(noisy_train_images[sample], clean_train_keypoints[sample], axis, "Random Noise Augmentation")
print("Shape of final train_images: {}".format(np.shape(train_images)))

print("Shape of final train_keypoints: {}".format(np.shape(train_keypoints)))



if horizontal_flip:

    print("Horizontal Flip Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(flipped_train_images[i], flipped_train_keypoints[i], axis, "")

    plt.show()



if rotation_augmentation:

    print("Rotation Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(rotated_train_images[i], rotated_train_keypoints[i], axis, "")

    plt.show()

    

if brightness_augmentation:

    print("Brightness Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(altered_brightness_train_images[i], altered_brightness_train_keypoints[i], axis, "")

    plt.show()



if shift_augmentation:

    print("Shift Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(shifted_train_images[i], shifted_train_keypoints[i], axis, "")

    plt.show()

    

if random_noise_augmentation:

    print("Random Noise Augmentation: ")

    fig = plt.figure(figsize=(20,8))

    for i in range(10):

        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])

        plot_sample(noisy_train_images[i], clean_train_keypoints[i], axis, "")

    plt.show()
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout, Activation, MaxPooling2D

from keras.optimizers import Adam, SGD, RMSprop

from keras import regularizers

from keras.layers.advanced_activations import ReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D



model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

# model.add(BatchNormalization())

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(0.1))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(ReLU())

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
from keras import optimizers

adam =optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=adam, 

              loss='mse', 

              metrics=['accuracy'])
import keras

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint



checkpointer = ModelCheckpoint(filepath = 'best_model.hdf5', monitor='val_mae', verbose=1, save_best_only=True, mode='min')

hist = model.fit(train_images, train_keypoints,epochs=100,batch_size =64, validation_split=0.20, callbacks=[checkpointer])
# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.save('keypoint_model.h5')
from keras.models import load_model 



test_preds = model.predict(test_images)
fig = plt.figure(figsize=(20,16))

for i in range(15):

    axis = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])

    plot_sample(test_images[i], test_preds[i], axis, "")

plt.show()