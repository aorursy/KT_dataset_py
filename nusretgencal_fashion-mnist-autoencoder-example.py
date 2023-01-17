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

        

import warnings

warnings.filterwarnings("ignore")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Model

from keras.layers import Input, Dense

from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0

x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(len(x_train), x_train.shape[1:][0]*x_train.shape[1:][1]) # len, 28*28

x_test = x_test.reshape(len(x_test), x_test.shape[1:][0]*x_test.shape[1:][1])
input_layer = Input(shape=(784,))



#encode architecture

encode_layer1 = Dense(64, activation='relu')(input_layer)

encode_layer2 = Dense(32, activation='relu')(encode_layer1)

encode_layer3 = Dense(16, activation='relu')(encode_layer2)



latent_view   = Dense(10, activation='sigmoid')(encode_layer3)



#decode architecture

decode_layer1 = Dense(16, activation='relu')(latent_view)

decode_layer2 = Dense(32, activation='relu')(decode_layer1)

decode_layer3 = Dense(64, activation='relu')(decode_layer2)



output_layer  = Dense(784, activation = 'sigmoid')(decode_layer3)
autoencoder = Model(input_layer, output_layer)

autoencoder.summary()
from keras.callbacks import EarlyStopping



autoencoder.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

hist = autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, validation_data=(x_train, x_train), callbacks=[early_stopping])
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")

plt.plot(hist.history["val_loss"], label = "Val Loss")

plt.legend()

plt.show()
preds = autoencoder.predict(x_train)
from PIL import Image 

f, ax = plt.subplots(1,10)

f.set_size_inches(80, 40)

for i in range(10):

    ax[i].imshow(x_train[i].reshape(28, 28))

plt.show()

f, ax = plt.subplots(1,10)

f.set_size_inches(80, 40)

for i in range(10):

    ax[i].imshow(preds[i].reshape(28, 28))

plt.show()