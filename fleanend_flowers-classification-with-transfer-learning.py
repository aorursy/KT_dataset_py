import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

from skimage import data,io,filters

from sklearn import preprocessing

from keras import preprocessing

from keras.layers.core import Dense,Flatten,Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Input,concatenate

from keras.models import Model

import keras

from keras import applications



train = False
flower = cv2.cvtColor(cv2.imread("../input/flowers-recognition/flowers/flowers/daisy/100080576_f52e8ee070_n.jpg"), cv2.COLOR_BGR2RGB)

io.imshow(flower)
num_classes = 5

x = []

y = []

training_dir = "../input/flowers-recognition/flowers/flowers/"



floropedia = {'daisy':0, 'rose':1, 'tulip':2, 'sunflower':3, 'dandelion':4}

inverse_floropedia = {0:'daisy', 1:'rose', 2:'tulip', 3:'sunflower', 4:'dandelion'}



for flower in os.listdir(training_dir):

    for m in os.listdir(training_dir+flower):

        if m[-3:] != 'jpg':

            continue

        x.append(cv2.resize(cv2.imread(training_dir+flower+"/"+m), (256,256)))

        y_i = np.zeros(num_classes)

        y_i[floropedia[flower]] = 1

        y.append(y_i)
if train:

    mask = np.random.choice([True,False],len(x), p=(0.7,0.3))

    x_train = np.array(x)[mask]

    y_train = np.array(y)[mask]

    x_test = np.array(x)[~mask]

    y_test = np.array(y)[~mask]







    # Image Augmentation

    image_generator_train = preprocessing.image.ImageDataGenerator(

        rescale=1./255,

        rotation_range=15,

        width_shift_range=.15,

        height_shift_range=.15,

        zoom_range = [.5,2],

        horizontal_flip=True)



    image_generator_test = preprocessing.image.ImageDataGenerator(

        rescale=1./255,

        rotation_range=15,

        width_shift_range=.15,

        height_shift_range=.15,

        zoom_range = [.5,2],

        horizontal_flip=True)





    image_generator_train.fit(x_train, augment=True)

    image_generator_test.fit(x_test, augment=True)
if train:

    augmented_train = image_generator_train.flow(np.array(x_train),np.array(y_train))

    augmented_test = image_generator_test.flow(np.array(x_test),np.array(y_test))

    for i,image in enumerate(augmented_train):

        io.imshow(image[0][0])

        if i == 1:

            break
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (256, 256, 3))



for layer in model.layers[:-5]:

    layer.trainable = False

    

#new_model = Model(model.inputs, model.layers[11].output)



    

#Adding custom Layers

x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

predictions = Dense(num_classes, activation="softmax")(x)





# creating the final model 

model_final = Model(input = model.input, output = predictions)



# compile the model 

model_final.compile(loss = "categorical_crossentropy", optimizer =keras.optimizers.SGD(lr=0.001), metrics=["accuracy"])

model_final.summary()
if train:



    # Save model at each iteration

    checkpoint = keras.callbacks.ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # early stopping

    early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')



    epochs = 100



    # Train the model 

    model_final.fit_generator(

    augmented_train,

    steps_per_epoch = len(x_train)/32,

    epochs = epochs,

    validation_data = augmented_test,

    validation_steps = len(x_test)/32,

    callbacks = [checkpoint, early])

    model_final.save_weights('100epochs_weights.h5')

else:

    model_final.load_weights('../input/flowers-classification-with-transfer-learning/100epochs_weights.h5')

    model_final.save_weights('100epochs_weights.h5')

    model_final.save('complete_model.h5')
url = "https://vignette.wikia.nocookie.net/disney/images/f/fb/The_Enchanted_Rose.png/revision/latest?cb=20160707142124"

image = io.imread(url)



print(floropedia)



io.imshow(image)

pred = model_final.predict(np.expand_dims(cv2.resize(image,(256,256)),axis=0))

print(pred, inverse_floropedia[np.argmax(pred)])