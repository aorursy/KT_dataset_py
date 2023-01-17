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
import warnings

warnings.simplefilter("ignore",category=FutureWarning)
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
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline

from sklearn.model_selection import train_test_split



from scipy.stats import norm



import tensorflow as tf

import keras

from keras.layers import *

from keras.models import *

from keras.losses import *

from keras.callbacks import *

from keras.optimizers import *

from keras import metrics

from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano



K.clear_session()



np.random.seed(237)

print(keras.__version__) 
train = pd.read_csv('../input/digit-recognizer/train.csv')

# train_orig = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

# test_orig = pd.read_csv('../input/digit-recognizer/test.csv')



train.tail()

      
# test_orig['label'] = 11

# testCols = test_orig.columns.tolist()

# testCols = testCols[-1:] + testCols[:-1]

# test_orig = test_orig[testCols]

# combined.tail()

# 検証/テストサンプルとして5000枚のランダムな画像を用意する

# valid = combined.sample(n = 5000, random_state = 555)

# train_after = combined.loc[~combined.index.isin(valid.index)]



# 余分なデータを消去

# del train_orig, test_orig, combined



#valid.head()

X_train = train.drop(['label'], axis = 1)

# X_valid = test_orig.drop(['label'], axis = 1)



# labelを取得

y_train = train['label']

# y_valid = valid['label']

# X_train, X_test, y_train, y_test = train_test_split(x_train,Y_train,test_size=0.1,shuffle=True)



# 正規化を行う

X_train = X_train.astype('float32') / 255.

X_train = X_train.values.reshape(-1,28,28,1)



X_test = test.astype('float32') / 255.

X_test = X_test.values.reshape(-1,28,28,1)



# num_classes = 10



# y_train = keras.utils.to_categorical(y_train, num_classes)



# X_valid = X_valid.astype('float32') / 255.

# X_valid = X_valid.values.reshape(-1,28,28,1)

plt.figure(1)

plt.subplot(221)

plt.imshow(X_train[13][:,:,0])



plt.subplot(222)

plt.imshow(X_train[690][:,:,0])



plt.subplot(223)

plt.imshow(X_train[2375][:,:,0])



plt.subplot(224)

plt.imshow(X_test[3360][:,:,0])

plt.show()
def sampling(args):

    z_mean, z_log_var = args

    batch = K.shape(z_mean)[0]

    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

inputs = Input(shape=(28,28,1), name='encoder_input')

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

# x = MaxPooling2D((2, 2), padding='same')(x)

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# x = MaxPooling2D((2, 2), padding='same')(x)

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# x = MaxPooling2D((2, 2), padding='same',name='encoded')(x)

x = Conv2D(32, 3, activation="relu", padding="same")(inputs)

x = Conv2D(64, 3, activation="relu", strides=(2,2), padding="same")(x)

x = Conv2D(64, 3, padding='same', activation='relu')(x)

x = Conv2D(64, 3, padding='same', activation='relu')(x)

shape = K.int_shape(x)

x = Flatten()(x)

x = Dense(32, activation='relu')(x)



z_mean = Dense(latent_dim, name='z_mean')(x)

z_log_var = Dense(latent_dim, name='z_log_var')(x)



z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])



encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

encoder.summary()



latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

# x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)

# x = Reshape((shape[1], shape[2], shape[3]))(x)

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# x = UpSampling2D((2, 2))(x)

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# x = UpSampling2D((2, 2))(x)

# x = Conv2D(16, (3, 3), activation='relu')(x)

# x = UpSampling2D((2, 2))(x)

# x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)

# x = Reshape((7, 7, 64))(x)

# x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)

# x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

# outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

# outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

x = Dense(np.prod(shape[1:]),activation='relu')(latent_inputs)

x = Reshape(shape[1:])(x)

x = Conv2DTranspose(32, 3,padding='same',activation='relu',strides=(2, 2))(x)

outputs = Conv2D(1, 3,padding='same', activation='sigmoid')(x)



# instantiate decoder model

decoder = Model(latent_inputs, outputs, name='decoder')

decoder.summary()



# instantiate VAE model

outputs = decoder(encoder(inputs)[2])

vae = Model(inputs, outputs, name='vae')
# Compute VAE loss

reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

reconstruction_loss *= 28 * 28

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)

kl_loss = K.sum(kl_loss, axis=-1)

kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop')

history=vae.fit(X_train,shuffle=True,epochs=10, batch_size=16,validation_data=(X_train, None))
print(type(history))



print(type(history.history))



print(history.history.keys())
a = vae.evaluate(X_test, y=None)

print(a)
def plot_label_clusters(encoder, decoder, data, labels):

    z_mean, _, _ = encoder.predict(data)

    plt.figure(figsize=(12, 10))

    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)

    plt.colorbar()

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.show()



X_train = np.expand_dims(X_train, -1).astype("float32") / 255



plot_label_clusters(encoder, decoder, X_train, y_train)
def plot_latent(input_img, y):

    # display a n*n 2D manifold of digits

    n = 30

    digit_size = 28

    scale = 2.0

    figsize = 15

    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = np.linspace(-scale, scale, n)

    grid_y = np.linspace(-scale, scale, n)[::-1]



    for i, yi in enumerate(grid_y):

        for j, xi in enumerate(grid_x):

            z_sample = np.array([[xi, yi]])

            x_decoded = decoder.predict(z_sample)

            digit = x_decoded[0].reshape(digit_size, digit_size)

            figure[

                i * digit_size : (i + 1) * digit_size,

                j * digit_size : (j + 1) * digit_size,

            ] = digit



    plt.figure(figsize=(figsize, figsize))

    start_range = digit_size // 2

    end_range = n * digit_size + start_range + 1

    pixel_range = np.arange(start_range, end_range, digit_size)

    sample_range_x = np.round(grid_x, 1)

    sample_range_y = np.round(grid_y, 1)

    plt.xticks(pixel_range, sample_range_x)

    plt.yticks(pixel_range, sample_range_y)

    plt.xlabel("z[0]")

    plt.ylabel("z[1]")

    plt.imshow(figure, cmap="Greys_r")

    plt.show()





plot_latent(encoder, decoder)
print(X_test.shape)



predictions = vae.predict(X_test)



print(type(predictions))



print(predictions.shape)



print(predictions)



print(predictions.sum())
results = np.argmax(predictions,axis = 1)

results = results.flatten()



results = results[results<10]

results = results[0:28000]

results = pd.Series(results,name="Label")

print(results)

print(type(results))

print(results.shape)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageID"),results],axis = 1)



submission.to_csv("vae_submission.csv",index=False)

submission