# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/boneage-training-dataset.csv")

train_df.shape
#print(os.listdir("../input/boneage-training-dataset/boneage-training-dataset"))

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



for img_id, boneage, male in train_df[['id','boneage','male']].sample(3).values:

    img_name = str(img_id) + '.png'

    img = mpimg.imread("../input/boneage-training-dataset/boneage-training-dataset/"+img_name)

    plt.imshow(img)

    plt.title('Image: {} Boneage: {} Male: {}'.format(img_name, boneage, male))

    plt.show()
import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



def read_and_prep_images(img_paths, img_height, img_width, color_mode="grayscale"):

    imgs = [load_img(img_path, target_size=(img_height, img_width), color_mode=color_mode) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    #output = preprocess_input(img_array)

    return(img_array)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



# quadratic approach

quadratic = True

image_size = 128



# some kind of ratio if not quadratic approach

img_rows = image_size if quadratic else 144 

img_cols = image_size if quadratic else 114



initial_model = Sequential()



# Input Layer

initial_model.add(Conv2D(12, kernel_size=(4,4), activation='relu', input_shape=(img_rows, img_cols, 1)))





# Convulotional Layers

initial_model.add(Conv2D(20, kernel_size=(4,4), activation='relu'))

initial_model.add(Conv2D(20, kernel_size=(3,3), activation='relu'))



# Flattening

initial_model.add(Flatten())



# Dense Layer

initial_model.add(Dense(100))

initial_model.add(Dense(1))



initial_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout



boner_model = Sequential()



# Input Layer

boner_model.add(Conv2D(16, kernel_size=(7,7), strides=(2,2), activation='relu', input_shape=(img_rows, img_cols, 3)))

boner_model.add(Dropout(0.25))



# Convulotional Layers

boner_model.add(Conv2D(32, kernel_size=(5,5), strides=(2,2), activation='relu'))

boner_model.add(Dropout(0.25))

boner_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

#boner_model.add(Dropout(0.25))

boner_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))

boner_model.add(Dropout(0.25))



# Flattening

boner_model.add(Flatten())



# Dense Layer

boner_model.add(Dense(256, activation='relu'))

#boner_model.add(Dropout(0.25))

boner_model.add(Dense(1))



boner_model.compile(loss='mean_squared_error', optimizer='adam')
from keras.preprocessing.image import ImageDataGenerator





train_df['imagepath'] = [f'{pid}.png' for pid in train_df.id]

#Lazy way of creating training and test set.

train = train_df.head(10000)

test = train_df.tail(2600)





datagen=ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,

                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

                           horizontal_flip=True, fill_mode="nearest")

train_generator=datagen.flow_from_dataframe(dataframe=train, 

                                            directory="../input/boneage-training-dataset/boneage-training-dataset/", 

                                            x_col="imagepath", 

                                            y_col="boneage", 

                                            class_mode="other", 

                                            target_size=(img_rows,img_cols), 

                                            batch_size=32)



valid_generator=datagen.flow_from_dataframe(dataframe=test, 

                                            directory="../input/boneage-training-dataset/boneage-training-dataset/", 

                                            x_col="imagepath", 

                                            y_col="boneage", 

                                            class_mode="other", 

                                            target_size=(img_rows,img_cols), 

                                            batch_size=32)



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = boner_model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=50)
import matplotlib.pyplot as plt

# Plot training & validation loss values

f, ax = plt.subplots(1,1, figsize=(15,10))

ax.plot(history.history['loss'])

ax.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

ax.grid(color='grey')

plt.show()
from tensorflow.keras import models

import torch

import cv2



layer_level = 5

filter = 3



def mygawd(layer_level, filter):

    opt_steps = 100

    upscaling_steps = 12

    upscaling_factor = 1.2

    blur = 5

    sz = 56

    

    img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255





    layer_outputs = [layer.output for layer in boner_model.layers] # Extracts the outputs of the top layers

    activation_model = models.Model(inputs=boner_model.input, outputs=layer_outputs[layer_level]) # Creates a model that will return these outputs, given the model input

    activation_model.trainable = False

    

    img_var = torch.tensor(np.array([img]),requires_grad=True)

    #print(f'img_var.mean() before: {img_var.mean()}')

    for _ in range(upscaling_steps):  # scale the image up upscaling_steps times

        optimizer = torch.optim.Adam([img_var], lr=0.1, weight_decay=1e-6)

        #print('peng')

        for n in range(opt_steps):  # optimize pixel values for opt_steps times

            #optimizer.zero_grad()

            # Ugly way to get the data out of the variable and make it posssible to send it to the model

            temp = img_var[:,:,:,:].detach().numpy()

            feature_map = activation_model.predict(cv2.resize(temp.reshape((sz,sz,3)), (128, 128)).reshape((1,128,128,3)))

            activation =torch.tensor(feature_map[0,:,:,filter],requires_grad=True)

            loss = -activation.mean()

            #print(f'loss: {loss}')

            loss.backward()

            # Couldn't figure out another way to update the pixels of img_var... probably won't work... sigh

            img_var.backward(gradient=torch.ones([1, sz, sz, 3], dtype=torch.float64) * activation.grad.mean())

            optimizer.step()

        sz = int(upscaling_factor * sz)  # calculate new image size

        img = img_var.data.cpu().numpy()[0]

        img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up

        if blur is not None: img = cv2.blur(img,(blur,blur)) # blur image to reduce high frequency patterns

        img_var = torch.tensor(np.array([img]),requires_grad=True)

    #print(f'img_var.mean() after: {img_var.mean()}')

    #print(f'activation.shape: {activation.shape}')

    #fig, ax = plt.subplots()

    #ax.imshow((img_var[0,:,:,].detach().numpy()* 255).astype(np.uint8))

    #ax.set_title(f'Layer {layer_level} Filter {filter}')

    #plt.axis('off')

    return (img_var[0,:,:,].detach().numpy()* 255).astype(np.uint8)



for i, filter in enumerate(range(100,110)):

    # Ad new row if row is full

    if i%5 == 0:

        fig, axes = plt.subplots(1,5, figsize=(20,5))

    filter_img = mygawd(layer_level, filter)

    axes[i%5].imshow(filter_img)

    at = axes[i%5].set_title(f'Layer {layer_level} Filter {filter}')

    at = axes[i%5].set_xticks([])

    at = axes[i%5].set_yticks([])