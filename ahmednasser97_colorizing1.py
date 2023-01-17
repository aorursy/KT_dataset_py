# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/testingh/test/Test/"))



# Any results you write to the current directory are saved as output.
from tensorflow.python.client import device_lib

from keras import backend as K



import os, sys, threading



import numpy as np

import tensorflow as tf



import keras

from keras.utils import multi_gpu_model

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.layers.core import RepeatVector, Permute

from keras.models import Model

from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate

from keras.initializers import TruncatedNormal

from keras.optimizers import RMSprop



from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb

from skimage.transform import resize

from skimage.io import imsave



import matplotlib.pyplot as plt


def batch_apply(ndarray, func, *args, **kwargs):

    """Calls func with samples, func should take ndarray as first positional argument"""



    batch = []

    for sample in ndarray:

        batch.append(func(sample, *args, **kwargs))

    return np.array(batch)

inception = InceptionResNetV2(weights='imagenet', include_top=True)

inception.graph = tf.get_default_graph()


def create_inception_embedding(grayscaled_rgb):

    '''Takes (299, 299, 3) RGB and returns the embeddings(predicions) generated on the RGB image'''

    with inception.graph.as_default():

        embed = inception.predict(grayscaled_rgb)

    return embed
def build_model():

    embed_input = Input(shape=(1000,))

    encoder_input = Input(shape=(256, 256, 1,))



    #Encoder

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2,

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_input)

    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)

    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2,

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)

    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)

    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2,

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)

    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)

    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)

    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(encoder_output)



    #Fusion

    fusion_output = RepeatVector(32 * 32)(embed_input)

    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)

    fusion_output = concatenate([encoder_output, fusion_output], axis=3)

    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(fusion_output)



    #Decoder

    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(fusion_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)

    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)

    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same',

                            bias_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)



    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    

    return model
datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)



# Convert images to LAB format and resizes to 256 x 256 for Encoder input.

# Also, generates Inception-resnet embeddings and returns the processed batch



def process_images(rgb, input_size=(256, 256, 3), embed_size=(299, 299, 3)):

    """Takes RGB images in float representation and returns processed batch"""



    # Resize for embed and Convert to grayscale

    gray = gray2rgb(rgb2gray(rgb))

    gray = batch_apply(gray, resize, embed_size, mode='constant')

    # Zero-Center [-1, 1]

    gray = gray * 2 - 1

    # Generate embeddings

    embed = create_inception_embedding(gray)



    # Resize to input size of model

    re_batch = batch_apply(rgb, resize, input_size, mode='constant')

    # RGB => L*a*b*

    re_batch = batch_apply(re_batch, rgb2lab)



    # Extract L* into X, zero-center and normalize

    X_batch = re_batch[:,:,:,0]

    X_batch = X_batch/50 - 1

    X_batch = X_batch.reshape(X_batch.shape+(1,))



    # Extract a*b* into Y and normalize. Already zero-centered.

    Y_batch = re_batch[:,:,:,1:]

    Y_batch = Y_batch/128



    return [X_batch, embed], Y_batch

def image_a_b_gen(images, batch_size):

    while True:

        for batch in datagen.flow(images, batch_size=batch_size):

            print("batch proccessed")

            yield process_images(batch)
DATASET = "../input/cliffabbeynature2/cliffabbeynature2/CliffAbbeyNature2/"

from tqdm import tqdm

# Get images file names

training_files, testing_files = train_test_split(shuffle(os.listdir(DATASET)), test_size=0.03)

print(len(training_files))

def getImages(DATASET, filelist, transform_size=(299, 299, 3)):

    """Reads JPEG filelist from DATASET and returns float represtation of RGB [0.0, 1.0]"""

    img_list = []

    i=0

    for filename in tqdm(filelist):

        # Loads JPEG image and converts it to numpy float array.

        image_in = img_to_array(load_img(DATASET + filename))



        # [0.0, 255.0] => [0.0, 1.0]

        image_in = image_in/255

        #print(i)

        i+=1

        if transform_size is not None:

            image_in = resize(image_in, transform_size, mode='reflect')



        img_list.append(image_in)

    img_list = np.array(img_list)



    return img_list
model = build_model()

#model.load_weights('../input/model1/mixed1/')



model.compile(optimizer=RMSprop(lr=1e-3), loss='mse',metrics=['accuracy'])


def train(model, training_files, batch_size=8, epochs=500, steps_per_epoch=100):

    print('Trains the model')

    training_set = getImages(DATASET, training_files)

    train_size = int(len(training_set)*0.90)

    

    train_images = training_set[:train_size]

    val_images = training_set[train_size:]

    val_steps = (len(val_images)//batch_size)

    print("Training samples:", train_size, "Validation samples:", len(val_images))



    callbacks = [

        EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-5),

        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, cooldown=0, verbose=1, min_lr=1e-8),

        ModelCheckpoint(monitor='val_acc', filepath='Mixed13.hdf5', verbose=1,

                         save_best_only=True, save_weights_only=True, mode='auto')

    ]



    history = model.fit_generator(image_a_b_gen(train_images, batch_size), epochs=epochs,

                                 steps_per_epoch=steps_per_epoch,

                        verbose=1, callbacks=callbacks, validation_data=process_images(val_images))

    

    return history
history = train(model, training_files, epochs=100)


import csv



def writeToCsv(acc_lst, loss_lst, val_acc_lst, val_loss_lst):    

   

    filename = 'training_metrics.csv'



    isOld = os.path.exists(os.path.join(filename))   

    

    with open (filename, 'a', newline='') as fp:

        a = csv.writer(fp)      

        if isOld == False:

            row = ['Accuracy', 'Loss', 'Val_Accuracy', 'Val_Loss']

            a.writerow(row)

        for acc, loss, val_acc, val_loss in zip(acc_lst, loss_lst, val_acc_lst, val_loss_lst):              

            row = [str(acc), str(loss), str(val_acc), str(val_loss)]

            a.writerow(row)

            

            

def plot_training_history(history):

    # Get the classification accuracy and loss-value

    # for the training-set.

    acc = history.history['acc']

    loss = history.history['loss']

    

    # Get it for the validation-set (we only use the test-set).

    val_acc = history.history['val_acc']

    val_loss = history.history['val_loss']



    writeToCsv(acc, loss, val_acc, val_loss)

    

    # Plot the accuracy and loss-values for the training-set.

    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')

    plt.plot(loss, 'o', color='b', label='Training Loss')

    

    # Plot it for the test-set.

    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')

    plt.plot(val_loss, 'o', color='r', label='Test Loss')



    # Plot title and legend.

    plt.title('Training and Test Accuracy')

    plt.legend()



    # Ensure the plot shows correctly.

    plt.show()
plot_training_history(history)
def test(model, training_files, save_actual=False, save_gray=False):

    test_images = getImages(DATASET, training_files)

    

    act = getImages(DATASET, training_files)

    #act = act*255

    model.load_weights(filepath="Mixed13.hdf5")



    print('Preprocessing Images')

    X_test, Y_test = process_images(test_images)



    print('Predicting')

    # Test model

    output = model.predict(X_test)



    # Rescale a*b* back. [-1.0, 1.0] => [-128.0, 128.0]

    output = output * 128

    Y_test = Y_test * 128

    pred = []



    # Output colorizations

    for i in tqdm(range(len(output))):

        #name = testing_files[i].split(".")[0]

        #print('Saving '+str(i)+"th image " + name + "_*.png")



        lightness = X_test[0][i][:,:,0]



        #Rescale L* back. [-1.0, 1.0] => [0.0, 100.0]

        lightness = (lightness + 1) * 50



        predicted = np.zeros((256, 256, 3))

        predicted[:,:,0] = lightness

        predicted[:,:,1:] = output[i]

        pred.append(predicted)

        

    return pred , act

      
pr, a= test(model, testing_files)



import cv2



fig, ax = plt.subplots(4, 3, figsize=(12,12))



pr[78]=lab2rgb(pr[78])

gray = gray2rgb(rgb2gray(a[78]))

ax[0,0].imshow(gray)

ax[0,1].imshow(pr[78])

ax[0,2].imshow(a[78])



pr[79]=lab2rgb(pr[79])

gray = gray2rgb(rgb2gray(a[79]))

ax[1,0].imshow(gray)

ax[1,1].imshow(pr[79])

ax[1,2].imshow(a[79])





pr[80]=lab2rgb(pr[80])

gray = gray2rgb(rgb2gray(a[80]))

ax[2,0].imshow(gray)

ax[2,1].imshow(pr[80])

ax[2,2].imshow(a[80])



pr[81]=lab2rgb(pr[81])

gray = gray2rgb(rgb2gray(a[81]))

ax[3,0].imshow(gray)

ax[3,1].imshow(pr[81])

ax[3,2].imshow(a[81])



from tqdm import tqdm

import cv2 

import numpy as np

names = os.listdir('../input/testingh/test/Test/')

GS_images = []

for i in tqdm(names):

    path = os.path.join('../input/testingh/test/Test/',i)

    im = cv2.imread(path)

    GS_images.append(im)

    

GS_images = np.array(GS_images)



def Testing(GS_images,embed_size = (299,299,3) , input_size = (256 ,256 ,3)):



    model.load_weights(filepath="Mixed12.hdf5")

    

    gray1 = batch_apply(GS_images, resize, embed_size, mode='constant')#299

    gray1 = gray1 * 2 - 1

    embed = create_inception_embedding(gray1)

    

    gray2 = batch_apply(GS_images, resize, input_size, mode='constant')#256

    gray2/50-1

    

    X_test = [gray2 , embed]

    output = model.predict(X_test)



    # Rescale a*b* back. [-1.0, 1.0] => [-128.0, 128.0]

    output = output * 128

    pred = []

    for i in tqdm(range(len(output))):



        lightness = X_test[0][i][:,:,0]



        #Rescale L* back. [-1.0, 1.0] => [0.0, 100.0]

        lightness = (lightness + 1) * 50



        predicted = np.zeros((256, 256, 3))

        predicted[:,:,0] = lightness

        predicted[:,:,1:] = output[i]

        pred.append(predicted)

    return pred
res = Testing(GS_images)

print(len(res))