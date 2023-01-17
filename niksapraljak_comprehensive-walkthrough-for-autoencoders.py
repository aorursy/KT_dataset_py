# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import os
import tensorflow as tf

tf.__version__

device_name = tf.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))
tf.test.is_gpu_available()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.metrics import Precision, Recall, AUC

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation

from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
main_path = '../input/digit-recognizer/'

train_df = pd.read_csv(main_path + 'train.csv')

test_df = pd.read_csv(main_path + 'test.csv')
i = 3*32

img_size = (28*28,)

X, y = np.zeros((i,) + img_size, dtype = "float32"), np.zeros((i,) + img_size, dtype = "float32")

for sample in range(i):

    img = train_df.iloc[sample,1:]

    X[ sample,:] = img

    y[ sample,:] = img
img.shape, X.shape, X.reshape(96,784,).shape
# split the training and testing dataframes    

def split_train_DF(df, train_perc):

    # train_perc --> the percentage in the training set

    final_train_df = df.iloc[0:round(train_perc*len((df.label))),:]

    val_df = df.iloc[round(train_perc*len((df.label))):,:]

    

    return final_train_df, val_df    





# catch statement: checks and verifies if the split was correct and 

# information/data was lost    

def verify_traintest_split(dataset, train_set, test_set):

    

    Total, train_A, train_B = len(dataset), len(train_set), len(test_set)

    if Total == (train_A + train_B): print('Splitting the dataset into testing and training is successful ...\n')

    else: print('Splitting the dataset into testing and training failed ...')

    return
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt





train_perc = 0.8

new_train_df, val_df = split_train_DF(train_df, train_perc)

# check if the dataframe was properly split ... 

verify_traintest_split(train_df, new_train_df, val_df)

column_names = train_df.columns
from tensorflow.keras import layers

from tensorflow.keras.models import Model









def get_model(latent_space = 32):



    input_img = Input(shape = (28*28,))

    encoded = Dense(latent_space, activation = 'relu')(input_img)

    decoded = Dense(28*28, activation = 'sigmoid')(encoded)

    return Model(input_img, decoded)





perc_autoencoder = get_model()

opt = tf.keras.optimizers.SGD(learning_rate=0.0100000231231)

perc_autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

perc_autoencoder.summary()
train_df.iloc[:,0]
from tqdm import tqdm 

from sklearn.model_selection import train_test_split

import numpy as np





X = np.zeros((len(train_df), 28*28,), dtype='float32')

y = np.zeros((len(train_df), 1), dtype='float32')



for sample in tqdm(range(len(train_df))):

    X[sample,:] = train_df.iloc[sample,1:].values

    X[sample,:] *= 1/255.0

    y[sample,0] = train_df.iloc[sample, 0]

    

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 42)
train_history = perc_autoencoder.fit(X_train,X_train,  validation_data = (X_val,X_val), 

                     epochs = 200, batch_size = 250, verbose = 2, shuffle = True)
plt.figure(figsize = (12,7))

plt.plot(train_history.history['loss'], 'r', LineWidth = 3, alpha = 0.45, label = 'training loss')

plt.plot(train_history.history['val_loss'], 'b', LineWidth = 3, alpha = 0.45, label = 'validation loss')

plt.xlabel('Epochs', fontsize = 18)

plt.ylabel('Loss', fontsize = 18)

plt.title('Comparing  loss between training and validation datasets', fontsize = 18, fontweight = 'bold')

plt.legend(fontsize = 18)

plt.show()
import gc

import psutil



# Free up space

process = psutil.Process(os.getpid())

print('Before deleting training data:', process.memory_info().rss)

del X, X_train, X_val; gc.collect()

print('After deleting training data:', process.memory_info().rss)
X = np.zeros((len(train_df), 28, 28, 1))

y = np.zeros((len(train_df), 1))

for sample in tqdm(range(len(train_df))):

    X[sample,:,:,0] = train_df.iloc[sample,1:].values.reshape(28,28,)

    y[sample,0] = train_df.iloc[sample,0]

print('Final shape:', X.shape, y.shape)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 42)
from tensorflow.keras import layers

from tensorflow.keras.models import Model





def gen_DCNN_autoencoder():

    input_layer = Input(shape = (28,28,1))



    x = layers.Conv2D(8, (3,3), padding = 'same', activation = 'relu')(input_layer)

    b = layers.BatchNormalization()(x)

    x = layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(x)

    mp = layers.MaxPooling2D((2,2), padding = 'same')(b)

    x = layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')(mp)

    b = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(x)    



    encoder = layers.MaxPooling2D((2,2), padding = 'same', name = 'encoding_z-space')(b)





    t = layers.Conv2DTranspose(32, (3,3), strides = (2,2),  padding = 'same', activation = 'relu')(encoder)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(32, (3,3), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(16, (3,3),  strides = (2,2), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(8, (3,3), padding = 'same', activation = 'relu')(b)



    decoded = layers.Conv2D(1, (3,3), padding = 'same', activation = 'sigmoid')(t)



    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')



    return autoencoder
# normalize the data such that pixels are binned between [0., 1.]

X_train *= 1/255.0

X_val *= 1/255.0
autoencoder = gen_DCNN_autoencoder()

history = autoencoder.fit(X_train, X_train, validation_data = (X_val,X_val), epochs = 30, batch_size = 250, shuffle=True, verbose = 2)
plt.figure(figsize = (25,5))

plt.plot(history.history['loss'], 'r', LineWidth = 3, alpha = 0.45, label = 'training loss')

plt.plot(history.history['val_loss'], 'b', LineWidth = 3, alpha = 0.45, label = 'validation loss')

plt.xlabel('Epochs', fontsize = 18)

plt.ylabel('Loss', fontsize = 18)

plt.title('Comparing loss between training and validation datasets', fontsize = 18, fontweight = 'bold')

plt.legend(fontsize = 18)

plt.show()
for sample in range(10):

    plt.figure(figsize = (5,5))

    plt.subplot(1,2,1)

    plt.imshow(X_val[sample].reshape(28,28,), cmap = 'gray')

    plt.title("Original image", fontsize = 18, fontweight = 'bold')

    plt.axis('off')



    plt.subplot(1,2,2)

    plt.imshow(autoencoder.predict(X_val[sample].reshape(1,28,28,1)).reshape(28,28,), cmap = 'gray')

    plt.title("Model prediction", fontsize = 18, fontweight = 'bold')

    plt.axis('off')

    plt.tight_layout()

    plt.show()
noise = np.random.uniform(0,1,(28,28,1))

plt.figure(figsize=(12,5))



plt.subplot(1,2,1)

plt.imshow(noise.reshape(28,28,), cmap = 'gray')

plt.title('Uniform distribution between [0,1]', fontsize = 18, fontweight = 'bold')

plt.axis('off')



plt.subplot(1,2,2)

plt.imshow(autoencoder.predict(noise.reshape(1,28,28,1)).reshape(28,28,), cmap = 'gray')

plt.title('Model predicted image', fontsize = 18, fontweight = 'bold')

plt.axis('off')



plt.tight_layout()

plt.show()


def Denoise_autoencoder(z):

    input_layer = Input(shape = (28,28,1))



    x = layers.Conv2D(8, (3,3), padding = 'same', activation = 'relu')(input_layer)

    b = layers.BatchNormalization()(x)

    mp = layers.MaxPooling2D((2,2), padding = 'same')(b)

    x = layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu')(mp)

    b = layers.BatchNormalization()(x)

    mp = layers.MaxPooling2D((2,2), padding = 'same')(b)

    x = layers.Conv2D(16, (3,3), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(x)    

    mp = layers.MaxPooling2D((2,2), padding = 'same')(b)

    x = layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')(mp)

    b = layers.BatchNormalization()(x)    

    mp = layers.MaxPooling2D((2,2), padding = 'same')(b)

    x = layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu')(mp)

    b = layers.BatchNormalization()(x)    

    mp = layers.MaxPooling2D((2,2), padding = 'same')(b)

    x = layers.Conv2D(z, (3,3), padding = 'same', activation = 'relu')(mp)

    b = layers.BatchNormalization()(x)    

    

    

    encoder = layers.MaxPooling2D((2,2), padding = 'same', name = 'encoding_z-space')(b)

    



    t = layers.Conv2DTranspose(64, (3,3), strides = (2,2),  padding = 'valid', activation = 'relu')(encoder)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(32, (3,3),  strides = (2,2), padding = 'valid', activation = 'relu')(b)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(16, (3,3),  strides = (2,2), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(16, (3,3),  strides = (2,2), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(t)

    t = layers.Conv2DTranspose(8, (3,3), padding = 'same', activation = 'relu')(b)

    b = layers.BatchNormalization()(t)



    

    

    decoded = layers.Conv2D(1, (3,3), padding = 'same', activation = 'sigmoid')(b)



    opt = tf.keras.optimizers.Adam(learning_rate=0.000100000231231)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')



    return autoencoder

denoise_ae = Denoise_autoencoder(3)

denoise_ae.summary()
import cv2 as cv

import random



sample = random.randint(0, 3000)

img = X_train[sample,:,:,0].copy()

noise_uni = cv.randu(img,(0),(1))



# Present the before and after adding noise ...

plt.imshow(X_train[sample,:,:,0].reshape(28,28,), cmap='gray')

plt.show()

plt.imshow((X_train[sample,:,:,0] + noise_uni), cmap='gray')

plt.show()
X_train_noise, X_val_noise = np.zeros((len(X_train), 28,28,1)), np.zeros((len(X_val), 28, 28, 1))



for sample in tqdm(range(len(X_train))):

    

    img = X_train[sample,:,:,0].copy()

    noise_img = cv.randu(img,(0),(0.35))

    X_train_noise[sample,:,:,0] =  X_train[sample,:,:,0] + noise_img

    

    if sample < len(X_val):

        img2 = X_val[sample,:,:,0].copy()

        noise_img2 = cv.randu(img2,(0),(0.35))

        X_val_noise[sample,:,:,0] =  X_val[sample,:,:,0] + noise_img2
plt.imshow(X_val[0,:,:,:].reshape(28,28,))

plt.show()

plt.imshow(X_val_noise[0,:,:,:].reshape(28,28,))

plt.show()



plt.imshow(X_train[0,:,:,:].reshape(28,28,))

plt.show()

plt.imshow(X_train_noise[0,:,:,:].reshape(28,28,))

plt.show()
denoise_ae = Denoise_autoencoder(3)

denoise_ae_history = denoise_ae.fit(X_train_noise, X_train, validation_data = (X_val_noise,X_val), epochs = 30, batch_size = 32, shuffle=True, verbose = 2)
alpha = random.randint(0, len(X_val))



plt.figure(figsize = (15,5))

plt.subplot(1,3,1)

plt.imshow(X_train[alpha,:,:,:].reshape(28,28,))

plt.title('original image')

plt.axis('off')

plt.subplot(1,3,2)

plt.imshow(X_train_noise[alpha,:,:,:].reshape(28,28,))

plt.title('+ noise')

plt.axis('off')

plt.subplot(1,3,3)

plt.imshow(denoise_ae.predict(X_train_noise[alpha,:,:,:].reshape(1,28,28,1)).reshape(28,28,))

plt.title('generated image')

plt.axis('off')

plt.show()







plt.figure(figsize = (15,5))

plt.subplot(1,3,1)

plt.imshow(X_val[alpha,:,:,:].reshape(28,28,))

plt.title('original image')

plt.axis('off')

plt.subplot(1,3,2)

plt.imshow(X_val_noise[alpha,:,:,:].reshape(28,28,))

plt.title('+ noise')

plt.axis('off')

plt.subplot(1,3,3)

plt.imshow(denoise_ae.predict(X_val_noise[alpha,:,:,:].reshape(1,28,28,1)).reshape(28,28,))

plt.title('generated image')

plt.axis('off')

plt.show()


encoder = Model(denoise_ae.input, denoise_ae.layers[-12].output)

encoder.summary()


train_latent_space = np.zeros((500, 3))

val_latent_space = np.zeros((500, 3))

for sample in tqdm(range(500)):

    train_latent_space[sample,:] = encoder.predict(X_train[sample,:,:,0].reshape(1,28,28,1))[0][0][0]

    val_latent_space[sample,:] = encoder.predict(X_val[sample,:,:,0].reshape(1,28,28,1))[0][0][0]




fig = plt.figure(figsize = (10, 7)) 

ax = plt.axes(projection ="3d") 

  

# Creating plot 

ax.scatter3D(train_latent_space[:,0], train_latent_space[:,1], train_latent_space[:,2], alpha = 0.4); 

ax.scatter3D(val_latent_space[:,0], val_latent_space[:,1], val_latent_space[:,2], alpha = 0.4); 

plt.title("simple 3D scatter plot") 

# show plot 

plt.show()



plt.figure(figsize = (12,3))

plt.subplot(1,3,1)

plt.scatter(train_latent_space[:,0], train_latent_space[:,1], alpha = 0.4)

plt.scatter(val_latent_space[:,0], val_latent_space[:,1], alpha = 0.4)

#plt.scatter(train_latent_space_8[:,0], train_latent_space_8[:,1])

plt.subplot(1,3,2)

plt.scatter(train_latent_space[:,0], train_latent_space[:,2], alpha = 0.4)

plt.scatter(val_latent_space[:,0], val_latent_space[:,2], alpha = 0.4)



#plt.scatter(train_latent_space_8[:,0], train_latent_space_8[:,2])

plt.subplot(1,3,3)

plt.scatter(train_latent_space[:,1], train_latent_space[:,2], alpha = 0.4)

plt.scatter(val_latent_space[:,1], val_latent_space[:,2], alpha = 0.4)



#plt.scatter(train_latent_space_8[:,1], train_latent_space_8[:,2])

plt.show()