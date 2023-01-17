import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from keras.preprocessing import image
train_images = sorted(os.listdir('/kaggle/input/medical-image-dataset/Dataset'))
train_image = []

for im in train_images:

    img = image.load_img('/kaggle/input/medical-image-dataset/Dataset/'+ im, target_size=(64,64), color_mode= 'grayscale')

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

train_df = np.array(train_image)
import matplotlib.pyplot as plt



def show_img(dataset):

    f, ax = plt.subplots(1,5)

    f.set_size_inches(40, 20)

    for i in range(5,10):

        ax[i-5].imshow(dataset[i].reshape(64,64), cmap='gray')

    plt.show()
def add_noice(image):

    row,col,ch= image.shape

    mean = 0

    sigma = 1

    gauss = np.random.normal(mean,sigma,(row,col,ch))

    gauss = gauss.reshape(row,col,ch)

    noisy = image + gauss*0.07

    return noisy
noised_df= []



for img in train_df:

    noisy= add_noice(img)

    noised_df.append(noisy)
noised_df= np.array(noised_df)
show_img(train_df)
show_img(noised_df)
noised_df.shape
train_df.shape
xnoised= noised_df[0:100]

xtest= noised_df[100:]
xnoised.shape
from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, MaxPooling2D,MaxPool2D ,UpSampling2D, Flatten, Input

from keras.optimizers import SGD, Adam, Adadelta, Adagrad

from keras import backend as K



def autoencoder():

    

    input_img = Input(shape=(64,64,1), name='image_input')

    

    #enoder 

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1')(input_img)

    x = MaxPooling2D((2,2), padding='same', name='pool1')(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2')(x)

    x = MaxPooling2D((2,2), padding='same', name='pool2')(x)

    

    #decoder

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3')(x)

    x = UpSampling2D((2,2), name='upsample1')(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv4')(x)

    x = UpSampling2D((2,2), name='upsample2')(x)

    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='Conv5')(x)

    

    #model

    autoencoder = Model(inputs=input_img, outputs=x)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    

    return autoencoder
model= autoencoder()

model.summary()
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
with tf.device('/device:GPU:0'):

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    model.fit(xnoised, xnoised, epochs=40, batch_size=10, validation_data=(xtest, xtest), callbacks=[early_stopping])
xtrain= train_df[100:]
import cv2



pred= model.predict(xtest[:5])

def plot_predictions(y_true, y_pred):    

    f, ax = plt.subplots(4, 5)

    f.set_size_inches(10.5,7.5)

    for i in range(5):

        ax[0][i].imshow(np.reshape(xtrain[i], (64,64)), aspect='auto', cmap='gray')

        ax[1][i].imshow(np.reshape(y_true[i], (64,64)), aspect='auto', cmap='gray')

        ax[2][i].imshow(np.reshape(y_pred[i], (64,64)), aspect='auto', cmap='gray')

        ax[3][i].imshow(cv2.medianBlur(xtrain[i], (5)), aspect='auto', cmap='gray')

       

    plt.tight_layout()

plot_predictions(xtest[:5], pred[:5])
new_image = cv2.medianBlur(xtrain[0], (5))

plt.figure(figsize=(6,3))

plt.subplot(121)

plt.imshow(pred[0].reshape(64,64), cmap='gray')

plt.title('Autoencoder Image')

plt.xticks([])

plt.yticks([])

plt.subplot(122)

plt.imshow(new_image, cmap='gray')

plt.title('Median Filter')

plt.xticks([])

plt.yticks([])

plt.show()
from math import log10, sqrt 

  

def PSNR(original, denoiced): 

    mse = np.mean((original - denoiced) ** 2) 

    if(mse == 0):  # MSE is zero means no noise is present in the signal . 

                  # Therefore PSNR have no importance. 

        return 100

    max_pixel = 255.0

    psnr = 20 * log10(max_pixel / sqrt(mse)) 

    return psnr 

  

value1 = PSNR(xtest[0], new_image)

value2 = PSNR(xtest[0], pred[0])



print(f"PSNR value for Denoised image is {value2} dB while for Median filtered image is {value1} dB")