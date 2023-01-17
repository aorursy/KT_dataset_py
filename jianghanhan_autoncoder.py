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
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://www.pyimagesearch.com/wp-content/uploads/2020/02/keras_denoising_autoencoder_header.png")



import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.layers as layers

from IPython.display import SVG

print(tf.__version__)

##查看版本
#分开数据集

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
##将数据分成28*28，加上将数据压缩成0-1之间

x_train = x_train.reshape((-1, 28*28)) / 255.0

x_test = x_test.reshape((-1, 28*28)) / 255.0



print(x_train.shape, ' ', y_train.shape)

print(x_test.shape, ' ', y_test.shape)


code_dim = 32

inputs = layers.Input(shape=(x_train.shape[1],), name='inputs')

code = layers.Dense(code_dim, activation='relu', name='code')(inputs)

outputs = layers.Dense(x_train.shape[1], activation='softmax', name='outputs')(code)



auto_encoder = keras.Model(inputs, outputs)

auto_encoder.summary()
##整个 autoencoder流程图

keras.utils.plot_model(auto_encoder, show_shapes=True)
#编码器图形

encoder = keras.Model(inputs,code)

keras.utils.plot_model(encoder, show_shapes=True)
#解码器图形

decoder_input = keras.Input((code_dim,))

decoder_output = auto_encoder.layers[-1](decoder_input)

decoder = keras.Model(decoder_input, decoder_output)

keras.utils.plot_model(decoder, show_shapes=True)
#定义训练方式

auto_encoder.compile(optimizer='adam',

                    loss='binary_crossentropy')
#训练模型

history = auto_encoder.fit(x_train, x_train, batch_size=64, epochs=100, validation_split=0.1)
##预测

encoded = encoder.predict(x_test)

decoded = decoder.predict(encoded)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))


n = 5

for i in range(n):

    ax = plt.subplot(2, n, i+1)

    plt.imshow(x_test[i].reshape(28,28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    ax = plt.subplot(2, n, n+i+1)

    plt.imshow(decoded[i].reshape(28,28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
##学习曲线

def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.grid(True)

    plt.gca().set_ylim(0, 1)

    plt.show()



plot_learning_curves(history)