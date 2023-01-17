# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm
from keras import backend as K
from keras.datasets import mnist
fashion_mnist=keras.datasets.fashion_mnist
(x_train,y_train),(x_test, y_test)=fashion_mnist.load_data()
print(len(x_train),len(y_train))
print(len(x_test),len(y_test))
np.random.seed(1000)
X_train = (x_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(60000, 784)
print(X_train.shape)
from keras.optimizers import Adam
adam = Adam(lr=0.0002, beta_1=0.5)
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU
input_dim = 10*10

generator = Sequential()

generator.add(Dense(256, input_dim=input_dim))
generator.add(LeakyReLU(0.2))

generator.add(Dense(512))
generator.add(LeakyReLU(0.2))

generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))

generator.add(Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy',
                  optimizer=adam)

generator.summary()


discriminator = Sequential()

discriminator.add(Dense(1024, input_dim=784))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy',
                      optimizer=adam)

discriminator.summary()
discriminator.trainable = False
ganInput = Input(shape=(input_dim,))
x = generator(ganInput)
ganOutput = discriminator(x)

gan = Model(input=ganInput, output=ganOutput)
gan.compile(loss='binary_crossentropy',
            optimizer=adam)
gan.summary()
def generate_and_plot():
  num_examples = 100
  noise = np.random.normal(0, 1, size=[num_examples, input_dim])
  generated_images = generator.predict(noise)
  generated_images = generated_images.reshape(num_examples, 28, 28)

  plt.figure(figsize=(10, 10))
  for i in range(num_examples):
    plt.subplot(10, 10, i+1)
    plt.imshow(generated_images[i],
               cmap='gray',
               interpolation='nearest')
    plt.axis('off')

  plt.show()

def train(epochs=1, batch_size=128):
  m = X_train.shape[0]
  batch_count = m // batch_size 
  
  generate_and_plot()
  
  
  for e in range(epochs):
    print(f'Epochs: {e}')
    for _ in tqdm(range(batch_count)):
      noise = np.random.normal(0, 1, size=[batch_size, input_dim])
      generated_images = generator.predict(noise)

      real_images = X_train[np.random.randint(0, m, size=batch_size)]

      X = np.concatenate([real_images, generated_images])
      y_dis = np.zeros(2*batch_size)
      y_dis[:batch_size] = 0.9

      discriminator.trainable = True
      d_loss = discriminator.train_on_batch(X, y_dis)

      discriminator.trainable = False


      noise = np.random.normal(0, 1, size=[batch_size, input_dim])
      y_gan = np.ones(batch_size)
      gan.train_on_batch(noise, y_gan)

    if e%10==0:
      generate_and_plot()

     
train(50)
generate_and_plot()
