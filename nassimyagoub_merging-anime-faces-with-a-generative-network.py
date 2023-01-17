import warnings

warnings.filterwarnings("ignore")



import keras

from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Reshape

from keras.callbacks import LearningRateScheduler

from keras.optimizers import Adam

from keras.layers import Conv2D, Dropout, Conv2DTranspose, UpSampling2D



import numpy as np

import os

import cv2

import matplotlib.pyplot as plt

from PIL import Image
def list_images(basePath, contains=None):

    # return the set of files that are valid

    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)



def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):

    # loop over the directory structure

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        # loop over the filenames in the current directory

        for filename in filenames:

            # if the contains string is not none and the filename does not contain

            # the supplied string, then ignore the file

            if contains is not None and filename.find(contains) == -1:

                continue



            # determine the file extension of the current file

            ext = filename[filename.rfind("."):].lower()



            # check to see if the file is an image and should be processed

            if ext.endswith(validExts):

                # construct the path to the image and yield it

                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")

                yield imagePath

                

def load_images(directory='', size=(64,64)):

    images = []

    labels = []  # Integers corresponding to the categories in alphabetical order

    label = 0

    

    imagePaths = list(list_images(directory))

    

    for path in imagePaths:

        

        if not('OSX' in path):

        

            path = path.replace('\\','/')



            image = cv2.imread(path) #Reading the image with OpenCV

            image = cv2.resize(image,size) #Resizing the image, in case some are not of the same size



            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    

    return images
images=np.array(load_images('../input'))
_,ax = plt.subplots(5,5, figsize = (8,8)) 

for i in range(5):

    for j in range(5):

        ax[i,j].imshow(images[5*i+j])

        ax[i,j].axis('off')
noise_size = 10000

epsilon = 0.00001 # Small float added to variance to avoid dividing by zero in the BatchNorm layers.

img_shape = (64, 64, 3)
# Create generative model



model = Sequential()



model.add(Dense(1024, activation='elu', input_shape=(noise_size,))) # (noise_size) -> (1024)

model.add(BatchNormalization())

model.add(Reshape((8,8,16))) # (8, 8, 16)

model.add(Dropout(0.1))



model.add(Conv2D(64, (3, 3), activation='elu', padding='same')) # (8, 8, 128)

model.add(BatchNormalization())

model.add(UpSampling2D((2, 2))) # (16, 16, 128)

model.add(Dropout(0.1))



model.add(Conv2D(64, (3, 3), activation='elu', padding='same')) # (16, 16, 64)

model.add(BatchNormalization())

model.add(UpSampling2D((2, 2))) # (32, 32, 64)

model.add(Dropout(0.1))



model.add(Conv2D(32, (3, 3), activation='elu', padding='same')) # (32, 32, 32)

model.add(BatchNormalization())

model.add(UpSampling2D((2, 2))) # (64, 64, 32)

model.add(Dropout(0.1))



model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same')) # (64, 64, 3)



model.compile(optimizer=Adam(lr=0.005), loss='binary_crossentropy')



model.summary()
# Fix random seed for reproductibility

np.random.seed(0)



# Selecting a random sample of noise_size images

idx = np.random.randint(0,len(images),noise_size)



# Scaling the data

train_y = images[idx,:,:,:]/255.



# Creating the identity matrix of size noise_size

train_X = np.zeros((noise_size,noise_size))

for i in range(noise_size):

    train_X[i,i] = 1
epochs = 300
def schedule(step):

    

    if step < 100:

        return 0.001

    

    elif step < 250:

        return 0.0001

    

    else:

        return 0.00003
scheduler = LearningRateScheduler(schedule, verbose=0)



model.fit(train_X, train_y, epochs=epochs, batch_size=128, callbacks=[scheduler], verbose=0)
# Recalling from memorized faces



for k in range(3):

    

    plt.figure(figsize=(15,1.5))

    

    for j in range(10):

    

        one_hot_input = np.zeros((noise_size))

        one_hot_input[np.random.randint(noise_size)] = 1

        

        img = model.predict(one_hot_input.reshape((-1, noise_size)))

        img = Image.fromarray((255*img).astype('uint8').reshape((64, 64, 3)))

        

        plt.subplot(1,10,j+1)

        

        plt.axis('off')

        plt.imshow(img)

        

    plt.show()
for k in range(10):

    a = np.random.randint(noise_size)

    b = np.random.randint(noise_size)

    print("Simple pixel average")

    plt.figure(figsize=(10,3))

    

    

    for j in range(10):

        input_vector = np.zeros((noise_size))

        

        proportion = j/9



        img = proportion*train_y[a] + (1-proportion)*train_y[b]

        img = Image.fromarray((255*img).astype('uint8').reshape((64,64,3)))

        

        plt.subplot(2,5,j+1)

        plt.axis('off')

        plt.imshow(img)

        

    plt.show()

    

    print("Merge with generative network")

    plt.figure(figsize=(10,3))

    

    for j in range(10):

        input_vector = np.zeros((noise_size))

        

        proportion = j/9

        # Percentage of the input coming from the first one-hot vector

        input_vector[a] = proportion

        # Percentage of the input coming from the second one-hot vector

        input_vector[b] = 1-proportion

        

        input_vector = input_vector/(np.sqrt(input_vector.dot(input_vector.T)))



        img = model.predict(input_vector.reshape((-1, noise_size)))

        img = Image.fromarray((255*img).astype('uint8').reshape((64,64,3)))

        

        plt.subplot(2,5,j+1)

        plt.axis('off')

        plt.imshow(img)

        

    plt.show()