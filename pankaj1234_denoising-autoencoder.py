import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from skimage import io

import os

%matplotlib inline





# Import the Convolutional Autoencoder utility script

from convolutionalautoencoder import CNAutoEnc

from tensorflow.keras.optimizers import Adam

        
from keras.preprocessing.image import load_img, img_to_array, array_to_img



def arrReshape(arr):

    arr = arr.astype("float32")/255.0

    return arr



def arrShape(arr):

    arr = (arr.astype("float32")*255.0).astype("uint8")

    return arr



def getImageArray(folderPath):

    for dirname, _, filenames in os.walk('/kaggle/input/denoise/denoising-dirty-documents/train/train'):

        images = np.ndarray(shape=(len(filenames), 320, 320, 1),dtype=np.float32)

        i=0

        for filename in filenames:

            # reference Tensorflow : https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img

            img = img_to_array(load_img(os.path.join(dirname,filename),color_mode="grayscale",target_size=(320,320), interpolation="nearest"))

            img= img.reshape((320,320,1))

            images[i] = img

            i+=1

    return images

trainNoise = arrReshape(getImageArray('/kaggle/input/denoise/denoising-dirty-documents/train/train'))

trainCleaned = arrReshape(getImageArray('/kaggle/input/denoise/denoising-dirty-documents/train_cleaned/train_cleaned'))

testNoise = arrReshape(getImageArray('/kaggle/input/denoise/denoising-dirty-documents/test/test'))
array_to_img(np.hstack([trainNoise[5],trainCleaned[5]]))
EPOCH=50

batchsize=5

(encoder,decoder, autoencoder) = CNAutoEnc.create((320,320,1))

autoencoder.compile(loss="mse",optimizer=Adam(lr=5e-4) )
history = autoencoder.fit(trainNoise,trainCleaned,epochs=EPOCH,batch_size=batchsize)
decodedImages = autoencoder.predict(testNoise)
sampleshow = None

for i in range(5):

    if sampleshow is None:

        sampleshow = np.hstack([decodedImages[i],testNoise[i]])

    else:

        sampleshow = np.vstack([sampleshow,np.hstack([decodedImages[i],testNoise[i]])])
array_to_img(sampleshow)
from tensorflow.keras.preprocessing.image import random_rotation

origImage = trainNoise[0]

junkImage  = random_rotation(origImage, rg=20, row_axis=0, col_axis=12, channel_axis=2, fill_mode='nearest', cval=0.0, interpolation_order=1)

junkImage = arrReshape(junkImage.reshape((320, 320, 1)))
array_to_img(junkImage)
anamolyPredict = autoencoder.predict(junkImage.reshape((1,320, 320, 1)))
mse = np.mean((origImage-anamolyPredict[0])**2)

print(mse)
array_to_img(anamolyPredict[0])