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
import tensorflow as tf

# import tensorflow_hub as hub

from datetime import datetime

import time

import pickle

import warnings

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.utils import shuffle

import cv2



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.linear_model import SGDClassifier

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

# from sklearn.cross_validation import StratifiedKFold 

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from PIL import Image

from tensorflow.keras.applications import VGG16

from tensorflow.keras.regularizers import l1

from tensorflow.keras import backend

from keras.utils import to_categorical

from tensorflow.keras.layers import Dense,GlobalAvgPool2D,Input,BatchNormalization,GlobalMaxPooling1D,GRU,LSTM,Activation,Bidirectional,TimeDistributed,Reshape

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Embedding,Dropout,SpatialDropout1D,Flatten,LeakyReLU,Conv2DTranspose

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.initializers import Constant

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam,SGD

from keras import backend as K



from tqdm import tqdm, trange
# read = lambda imname: np.asarray(Image.open(imname).convert(""))

read = lambda imname: mpimg.imread(imname)
folder_benign_train = '/kaggle/input/skin-cancer-malignant-vs-benign/train/benign/'

folder_malignant_train = '/kaggle/input/skin-cancer-malignant-vs-benign/train/malignant'



folder_benign_test = '/kaggle/input/skin-cancer-malignant-vs-benign/test/benign'

folder_malignant_test = '/kaggle/input/skin-cancer-malignant-vs-benign/test/malignant'
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]

X_malignant = np.array(ims_malignant, dtype='uint8')
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]

X_benign_test = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]

X_malignant_test = np.array(ims_malignant, dtype='uint8')

y_benign = np.zeros(X_benign.shape[0])

y_malignant = np.ones(X_malignant.shape[0])



y_benign_test = np.zeros(X_benign_test.shape[0])

y_malignant_test = np.ones(X_malignant_test.shape[0])





# Merge data 

X_train = np.concatenate((X_benign, X_malignant), axis = 0)

y_train = np.concatenate((y_benign, y_malignant), axis = 0)



X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)

y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)

X_train,y_train = shuffle(X_train,y_train)
X_test,y_test = shuffle(X_test,y_test)
y_train
plt.imshow(X_train[y_train==0][np.random.randint(0,10),:,:])

plt.figure(figsize=(5,50))
cols = 5

num_classes = 2

fig,axs = plt.subplots(nrows = num_classes,ncols = cols,figsize = (10,10))

for i in range(num_classes):

    for j in range(cols):

        if i == 0:

            axs[i][j].set_title("benign")

        else:

            axs[i][j].set_title("malignant")

        x_selected = X_train[y_train==i]

        axs[i][j].imshow(x_selected[np.random.randint(0,10),:,:])

        axs[i][j].axis("off")
def grayScale(img):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img
img = grayScale(X_train[10])

plt.imshow(img)
def equalizationOfImage(img):

#this func will equalize image which means the brightness of image is flattened

    return cv2.equalizeHist(img)
img = equalizationOfImage(img)

plt.imshow(img)
def imgPreprocessing(img):

    img = grayScale(img)

    img = equalizationOfImage(img)

    img = img/255

    return img
X_train1 = np.array(list(map(imgPreprocessing,X_train)))

X_test1 = np.array(list(map(imgPreprocessing,X_test)))
plt.imshow(X_train[100])

X_train1[100].shape
X_train2 = X_train1.reshape(X_train1.shape[0],X_train1.shape[1],X_train1.shape[2],1)

X_test2 = X_test1.reshape(X_test1.shape[0],X_test1.shape[1],X_test1.shape[2],1)
X_train2,X_validation2,y_train2,y_validation2 = train_test_split(X_train2,y_train,test_size=0.2,random_state=42)
X_train2.shape
def building_model():

    model = Sequential()

    model.add(Conv2D(32,(5, 5),input_shape=X_train2.shape[1:],activation="relu",kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(Conv2D(32,(5,5), activation = 'relu',kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(64,(5,5),activation="relu",kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(Conv2D(64,(5,5),activation="relu",kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),activation="relu",kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(Conv2D(128,(3,3),activation="relu",kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    model.add(Dense(128,activation="relu",kernel_initializer = "RandomNormal",bias_initializer="zeros"))

    model.add(Dropout(0.2))

    model.add(Dense(1,activation="sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.001),metrics=["accuracy"])

    return model
model = building_model()
filepath="/kaggle/input/kaggle/working/model.h5"



checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='max')

callbackList = [checkpoint]
history = model.fit(X_train2,y_train2,validation_data=(X_validation2,y_validation2),batch_size=210,epochs = 50,callbacks=callbackList)
generator = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,rotation_range=30,horizontal_flip=True,vertical_flip=True)
generator.fit(X_train2)
for X_batch, y_batch in generator.flow(X_train2, y_train2, batch_size=9):

    print(y_batch[5])

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(X_batch[i].reshape(224,224))

    plt.show()

    break
param = {

    'theta':30,

    'shear':30,

    'flip_vertical':True,

    'flip_horizontal':True

}
generated_img = np.array([generator.apply_transform(x,param) for x in X_train2])
generated_img.shape
plt.imshow(X_train2[1].reshape(224,224))
plt.imshow(generated_img[1].reshape(224,224))
X_train3 = np.concatenate([X_train2,generated_img])
X_train3.shape
y_train3 = np.concatenate([y_train2,y_train2])

y_train3.shape
X_train3,y_train3 = shuffle(X_train3,y_train3)
model = building_model()

filepath="/kaggle/output/kaggle/working/model.h5"



checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='max')

callbackList = [checkpoint]
model.fit(X_train3,y_train3,

          validation_data=(X_validation2,y_validation2),batch_size=210,epochs = 20,callbacks=callbackList,shuffle=True)
X_test3 = X_test1.reshape(X_test1.shape[0],X_test1.shape[1],X_test1.shape[2],1)

X_test3.shape
y_test3 = y_test.reshape(y_test.shape[0],1)
model.evaluate(X_test3,y_test3)
X_validation1 = np.array([generator.apply_transform(x,param) for x in X_validation2])
X_validation1.shape
model_gen = building_model()
filepath="model_gen.h5"



checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True, mode='max')

callbackList = [checkpoint]
history = model_gen.fit_generator(generator.flow(X_train2,y_train2),validation_data=(X_validation1,y_validation2),

                                  steps_per_epoch=X_train1.shape[0]/50,epochs = 50,callbacks=callbackList)

def make_descriminator(inputShape=(224,224,1)):

    model = Sequential()

    model.add(Conv2D(32,(5, 5),padding="same",input_shape=inputShape))

    model.add(LeakyReLU(0.01))

    model.add(Conv2D(64,(5,5)))

    model.add(LeakyReLU(0.01))

    model.add(MaxPooling2D((2,2)))

    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3)))

    model.add(LeakyReLU(0.01))

    model.add(Conv2D(128,(3,3)))

    model.add(LeakyReLU(0.01))

#     model.add(GlobalAvgPool2D())

    model.add(Flatten())

    model.add(Dense(1))

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])



    return model
def make_generator():

    model = Sequential()

    model.add(Dense(128*28*28,input_shape=(100,)))

    model.add(BatchNormalization())

    model.add(LeakyReLU(0.1))

    model.add(Reshape((28,28,128)))

#     assert model.output_shape == (None, 28, 28, 128)

    model.add(Conv2DTranspose(64,(4,4),padding="same",strides=(2,2)))

#     assert model.output_shape == (None, 56, 56, 64)

    model.add(BatchNormalization())

    model.add(LeakyReLU(0.1))

    model.add(Conv2DTranspose(32,(4,4),padding="same",strides=(2,2)))

#     assert model.output_shape == (None, 112, 112, 32)

    model.add(BatchNormalization())

    model.add(LeakyReLU(0.1))

    model.add(Conv2DTranspose(1,(4,4),activation="tanh",strides=(2,2),padding="same"))

#     assert model.output_shape == (None, 223, 224, 1)

    return model

    

              
generator = make_generator()

noise = tf.random.normal([1,100])

generated_img = generator(noise,training=False)

plt.imshow(generated_img[0, :, :, 0], cmap='gray')
discriminator = make_descriminator()

decision = discriminator(generated_img)

print(decision)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output,y_real,fake_output,y_fake):

    real_loss = cross_entropy(y_real,real_output)

    fake_loss = cross_entropy(y_fake, fake_output)

#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output),fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
noise_dim = 100

num_examples_to_generate = 8



# We will reuse this seed overtime (so it's easier)

# to visualize progress in the animated GIF)

seed = tf.random.normal([num_examples_to_generate, noise_dim])
def generate_fake_images(generator,noise,n_samples):

    generated_images = generator(noise,training=True)

    y = np.zeros((n_samples,1))

    return generated_images,y
def generate_real_images(real_images,labels,n_samples):

#     real_images,labels = dataset

    ix = np.random.randint(0,real_images.shape[0],n_samples)

    X,labels = real_images[ix],labels[ix]

    

    return X,labels
def generate_latent_points(latent_dim, n_samples):

    z_input = np.random.randn(latent_dim * n_samples)

    z_input = z_input.reshape(n_samples, latent_dim)

    return z_input
def train(X_train,y_train,noise_dim,epochs,n_batches = 100):

    

    batch_per_epochs = int(X_train.shape[0]/n_batches)

    n_steps = batch_per_epochs * epochs

    half_batch = int(n_batches / 2)

    

    for i in range(n_steps):

        noise = tf.random.normal([half_batch, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            X_real,y_real = generate_real_images(X_train,y_train,half_batch)

    #         d_loss,d_acc = d_model.train_on_batch(X_real,y_real)

            real_output = discriminator(X_real,training=True)

            X_fake,y_fake = generate_fake_images(generator,noise,half_batch)

            fake_output = discriminator(X_fake,training=True)

            

            d_loss = discriminator_loss(real_output,y_real,fake_output,y_fake)

            g_loss = generator_loss(fake_output)

            

            gradient_of_gen = gen_tape.gradient(g_loss,generator.trainable_variables)

            gradient_of_dis = disc_tape.gradient(d_loss,discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradient_of_gen,generator.trainable_variables))

        discriminator_optimizer.apply_gradients(zip(gradient_of_dis,discriminator.trainable_variables))

            



#             X_gan,y_gan = generate_latent_points(100,n_batches),np.ones((n_batches,1))

#             g_loss = gan_model.train_on_batch(X_gan,y_gan)



        if (i+1) % (batch_per_epochs * 1) == 0:

            print('>%d, c[%.3f], g[%.3f]' % (i+1, d_loss, g_loss))

y_train.shape
X_train4 = X_train1.reshape(X_train1.shape[0],X_train1.shape[1],X_train1.shape[2],1)

y_train4 = y_train.reshape(y_train.shape[0],1)

noise_DIM = 100

EPOCHS = 50
train(X_train=X_train4,y_train=y_train4,noise_dim=noise_DIM,epochs=EPOCHS)
y_test4 = y_test.reshape(y_test.shape[0],1)

y_test4.shape
X_test4 = X_test1.reshape(X_test1.shape[0],X_test1.shape[1],X_test1.shape[2],1)
X_test4.shape
discriminator.evaluate(X_test4,y_test4)