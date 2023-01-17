###############################################################################################################################

#

#	Variational Auto Encoder Built using Keras and celebA dataset is used 

#	for training.

#	

#	Created by : B V P Sai Kumar 

#	github : https://github.com/kumararduino

#	website : https://kumarbasaveswara.in

#	linkedin : https://www.linkedin.com/in/kumar15412304/

#

#

#	Credits:

#	Dataset : https://www.kaggle.com/jessicali9530/celeba-dataset

#	Article : https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

#		This Article gave me a clear glance about how Variational_Auto_Encoders work

# Article on KL_Divergence : https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained

#	

#

#

###############################################################################################################################
import gc

import psutil

import multiprocessing as mp

import copy

mp.cpu_count()

import numpy as np

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import keras

from keras.models import Model,Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Lambda,Activation,Input,Flatten,Reshape,Conv2DTranspose

import keras.backend as K

from keras.layers.merge import add

from sklearn.model_selection import train_test_split

import os

import glob

from time import time,asctime

from random import randint as r

import random
# os.chdir("faces")

imgs = glob.glob("../input/img_align_celeba/img_align_celeba/*.jpg")

print(len(imgs))
imgs[0]
train_y = []

train_y2 = []

for _ in range(0,100000):

  if _%20000 == 0:

    print("{} / 100000".format(_))

  img = cv2.imread(imgs[_])

  img = cv2.resize(img,(32,32),interpolation = cv2.INTER_AREA)

  train_y.append(img.astype("float32")/255.0)

for _ in range(100000,200000):

  if _%20000 == 0:

    print("{} / 200000".format(_))

  img = cv2.imread(imgs[_])

  img = cv2.resize(img,(32,32),interpolation = cv2.INTER_AREA)

  train_y2.append(img.astype("float32")/255.0)

train_y = np.array(train_y)

train_y2 = np.array(train_y2)

Y_data = np.vstack((train_y,train_y2))

print(psutil.virtual_memory())

del train_y,train_y2

gc.collect()

print(psutil.virtual_memory())

Z_data = copy.deepcopy(Y_data)

Z_data = (Z_data - Z_data.mean())/Z_data.std()
test_Y = []

for _ in range(200000,202599):

  if _%5000 == 0:

    print("{} / 100000".format(_))

  img = cv2.imread(imgs[_])

  img = cv2.resize(img,(32,32),interpolation = cv2.INTER_AREA)

  test_Y.append(img.astype("float32")/255.0)

  

test_Y = np.array(test_Y)

mean = test_Y.mean()

std = test_Y.std()

test_Z = (test_Y - mean)/std
def sampler(layers):

  std_norm = K.random_normal(shape=(K.shape(layers[0])[0], 128), mean=0, stddev=1)

  return layers[0] + layers[1]*std_norm
stride = 2

inp = Input(shape = (32,32,3))

x = inp

x = Conv2D(32,(2,2),strides = stride,activation = "relu",padding = "same")(x)

x = Conv2D(64,(2,2),strides = stride,activation = "relu",padding = "same")(x)

x = Conv2D(128,(2,2),strides = stride,activation = "relu",padding = "same")(x)

shape = K.int_shape(x)

x = Flatten()(x)

x = Dense(256,activation = "relu")(x)

mean_layer = Dense(128,activation = "relu")(x)

sd_layer = Dense(128,activation = "relu")(x)

latent_vector = Lambda(sampler)([mean_layer,sd_layer])

encoder = Model(inp,latent_vector,name = "VAE_Encoder")

encoder.summary()

decoder_inp = Input(shape = (128,))

x = decoder_inp

x = Dense(shape[1]*shape[2]*shape[3],activation = "relu")(x)

x = Reshape((shape[1],shape[2],shape[3]))(x)

x = (Conv2DTranspose(32,(3,3),strides = stride,activation = "relu",padding = "same"))(x)

x = (Conv2DTranspose(16,(3,3),strides = stride,activation = "relu",padding = "same"))(x)

x = (Conv2DTranspose(8,(3,3),strides = stride,activation = "relu",padding = "same"))(x)

outputs = Conv2DTranspose(3, (3,3), activation = 'sigmoid', padding = 'same', name = 'decoder_output')(x)

decoder = Model(decoder_inp,outputs,name = "VAE_Decoder")

decoder.summary()
autoencoder = Model(inp,decoder(encoder(inp)),name = "Variational_Auto_Encoder")

autoencoder.summary()
def vae_loss(input_img, output):

	# compute the average MSE error, then scale it up, ie. simply sum on all axes

	reconstruction_loss = K.sum(K.square(output-input_img))

	# compute the KL loss

	kl_loss = - 0.5 * K.sum(1 + sd_layer - K.square(mean_layer) - K.square(K.exp(sd_layer)), axis=-1)

	# return the average loss over all images in batch

	total_loss = K.mean(reconstruction_loss + kl_loss)    

	return total_loss
autoencoder.compile(optimizer = "adam",loss = vae_loss,metrics = ["accuracy"])
autoencoder.fit(Z_data,Y_data,batch_size = 200,epochs = 15,validation_split = 0.5)
autoencoder.fit(Z_data,Y_data,batch_size = 32,epochs = 30,validation_split = 0.5)
autoencoder.fit(Z_data,Y_data,batch_size = 150,epochs = 30,validation_split = 0)
autoencoder.fit(Z_data,Y_data,batch_size = 200,epochs = 500,validation_split = 0)
pred = autoencoder.predict(test_Z)
temp = r(0,2599)

print(temp)

plt.subplot(1,3,1)

plt.imshow(test_Y[temp])

plt.subplot(1,3,2)

plt.imshow(test_Z[temp])

plt.subplot(1,3,3)

plt.imshow(pred[temp])
gen = np.random.normal(size = (1,32,32,3))

gen_sample = autoencoder.predict(gen)

plt.subplot(1,2,1)

plt.imshow(gen[0])

plt.subplot(1,2,2)

plt.imshow(gen_sample[0])
autoencoder.save_weights("VAE-weights-"+str(r(0,3653))+".h5")
for _ in range(10):

    img = np.random.normal(size = (9,32,32,3))

    pred = autoencoder.predict(img)

    op = np.vstack((np.hstack((pred[0],pred[1],pred[2])),np.hstack((pred[3],pred[4],pred[5])),np.hstack((pred[6],pred[7],pred[8]))))

    print(op.shape)

    op = cv2.resize(op,(288,288),interpolation = cv2.INTER_AREA)

    cv2.imwrite("generated"+str(r(0,9999))+".jpg",(op*255).astype("uint8"))