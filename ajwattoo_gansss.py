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
import glob
train_images_alder = glob.glob('/kaggle/input/trunk-assignment/alder/*.JPG')
train_images_ginkgo_biloba = glob.glob('/kaggle/input/trunk-assignment/ginkgo biloba/*.JPG')
train_images_birch = glob.glob('/kaggle/input/trunk-assignment/birch/*.JPG')
train_images_beech = glob.glob('/kaggle/input/trunk-assignment/beech/*.JPG')
train_images_chestnut = glob.glob('/kaggle/input/trunk-assignment/chestnut/*.JPG')
train_images_hornbeam = glob.glob('/kaggle/input/trunk-assignment/hornbeam/*.JPG')
train_images_horse_chestnut = glob.glob('/kaggle/input/trunk-assignment/horse chestnut/*.JPG')
train_images_linden = glob.glob('/kaggle/input/trunk-assignment/linden/*.JPG')
train_images_oak = glob.glob('/kaggle/input/trunk-assignment/oak/*.JPG')
train_images_pine = glob.glob('/kaggle/input/trunk-assignment/pine/*.JPG')
train_images_spruce = glob.glob('/kaggle/input/trunk-assignment/spruce/*.JPG')
train_images_oriental_plane= glob.glob('/kaggle/input/trunk-assignment/oriental plane/*.JPG')
train_images=[train_images_alder,train_images_ginkgo_biloba,train_images_birch,train_images_beech,train_images_chestnut,train_images_hornbeam ,train_images_horse_chestnut,train_images_linden,train_images_oak,train_images_pine,train_images_spruce,train_images_oriental_plane]
import cv2

y_train=[]
X_train = []
for (label, fnames) in enumerate(train_images):
    for fname in fnames:
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128 ,128 ) , interpolation=cv2.INTER_AREA)
            img=  img.astype('float32') / 255.
            y_train.append(label)
            X_train.append(img)
import numpy as np
X_train=np.array(X_train)
print(X_train.shape)
categories=['alder','ginkgo_biloba','birch','beech','chestnut','hornbeam','horse_chestnut','linden','oak','pine','spruce','oriental_plane']
import matplotlib.pyplot as plt
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221,title=categories[y_train[i]]), plt.imshow(X_train[i], cmap='gray')
plt.subplot(222,title=categories[y_train[i+34]]), plt.imshow(X_train[i+34], cmap='gray')
plt.subplot(223,title=categories[y_train[i+200]]), plt.imshow(X_train[i+200], cmap='gray')
plt.subplot(224,title=categories[y_train[i+300]]), plt.imshow(X_train[i+300], cmap='gray')
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,train_size=0.8,shuffle=True, stratify=y_train)
from keras.utils.np_utils import to_categorical

Y_train= to_categorical(Y_train)
y_test=to_categorical(y_test)
print(x_train.shape)
print(x_test.shape)
print(Y_train.shape)
print(y_test.shape)
from tensorflow.keras.optimizers import Adam
import numpy as np
np.random.seed(10)

# The dimension of z
noise_dim = 100

batch_size = 16
steps_per_epoch = 3750 # 60000 / 16
epochs = 10

save_path = 'fcgan-images'

img_rows, img_cols, channels = 128, 128, 1

optimizer = Adam(0.0002, 0.5)
import os
if save_path != None and not os.path.isdir(save_path):
    os.mkdir(save_path)
def create_generator():
    generator = Sequential()
    
    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(img_rows*img_cols*channels, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator
def create_descriminator():
    discriminator = Sequential()
     
    discriminator.add(Dense(1024, input_dim=img_rows*img_cols*channels))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator
from keras.initializers import RandomNormal
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras import initializers
!pip install --upgrade tensorflow-gpu
from tensorflow.compat.v1.keras import backend as k
session =tf.compat.v1.Session(graph=tf.Graph())
with session.graph.as_default():
    k.set_session(session)
session = tf.compat.v1.keras.backend.get_session()
init=tf.compat.v1.global_variables_initializer()
session.run(init)
discriminator = create_descriminator()
generator = create_generator()

# Make the discriminator untrainable when we are training the generator.  This doesn't effect the discriminator by itself
discriminator.trainable = False

# Link the two models to create the GAN
gan_input = Input(shape=(noise_dim,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)
def show_images(noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        if channels == 1:
            plt.imshow(np.clip(image.reshape((img_rows, img_cols)), 0.0, 1.0), cmap='gray')
        else:
            plt.imshow(np.clip(image.reshape((img_rows, img_cols, channels)), 0.0, 1.0))
        plt.axis('off')
    
    plt.tight_layout()
    
    if epoch != None and save_path != None:
        plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')
    plt.show()
x_train = x_train.reshape(-1, img_rows*img_cols*channels)
print(x_train.shape)
import matplotlib.pyplot as plt
def show_images(noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        if channels == 1:
            plt.imshow(image.reshape((img_rows, img_cols)), cmap='gray')
        else:
            plt.imshow(image.reshape((img_rows, img_cols, channels)))
        plt.axis('off')
    
    plt.tight_layout()
    
    if epoch != None and save_path != None:
        plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')
    plt.show()
static_noise = np.random.normal(0, 1, size=(100, noise_dim))
for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_x = generator.predict(noise)

        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        
        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9

        d_loss = discriminator.train_on_batch(x, disc_y)

        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
    if epoch % 5 == 0:
        show_images(static_noise, epoch)



noise = np.random.normal(0, 1, size=(100, noise_dim))
show_images(noise)
