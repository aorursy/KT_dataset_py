import os

import gc

import random

import json

import cv2

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from pathlib import Path



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.model_selection import train_test_split,KFold



from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, UpSampling1D, UpSampling2D, Lambda, Embedding, Flatten, Add,Concatenate, Dropout, LSTM

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Nadam

import keras.backend as K



from keras.applications.vgg16 import VGG16
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))

print(len(training_tasks), len(evaluation_tasks), len(test_tasks))
def get_data(task_filename):

    with open(task_filename, 'r') as f:

        task = json.load(f)

    return task



num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

color2num = {c: n for n, c in enumerate(num2color)}
for i in range(400):

    print(get_data(str(test_path / test_tasks[i])))

    break
for i in range(400):

    print(get_data(str(training_path / training_tasks[i]))['test'])

    break
x_train = []

y_train = []

x_test = []

y_test = []



ox_train = []

oy_train = []

ox_test = []



for i in range(400):

    for train_data in get_data(str(training_path / training_tasks[i]))['train']:

        x_train.append(cv2.resize(np.asarray(train_data['input']), dsize=(32, 32), interpolation=cv2.INTER_NEAREST))

        y_train.append(cv2.resize(np.asarray(train_data['output']), dsize=(32, 32), interpolation=cv2.INTER_NEAREST))

        ox_train.append(np.asarray(train_data['input']))

        oy_train.append(np.asarray(train_data['output']))

        

for i in range(100):

    for test_data in get_data(str(test_path / test_tasks[i]))['test']:

        x_test.append(cv2.resize(np.asarray(test_data['input']), dsize=(32, 32), interpolation=cv2.INTER_NEAREST))

        ox_test.append(np.asarray(test_data['input']))

    for train_data in get_data(str(test_path / test_tasks[i]))['train']:

        x_train.append(cv2.resize(np.asarray(train_data['input']), dsize=(32, 32), interpolation=cv2.INTER_NEAREST))

        y_train.append(cv2.resize(np.asarray(train_data['output']), dsize=(32, 32), interpolation=cv2.INTER_NEAREST))

        ox_train.append(np.asarray(train_data['input']))

        oy_train.append(np.asarray(train_data['output']))

        

x_train = np.asarray(x_train) / 10. 

y_train = np.asarray(y_train) / 10.

x_test = np.asarray(x_test) / 10.



print('length of x_train:', len(x_train))

print('length of x_test:', len(x_test))
#ref: https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img):

        img_h, img_w, img_c = input_img.shape

        p_1 = np.random.rand()



        if p_1 > p:

            return input_img



        while True:

            s = np.random.uniform(s_l, s_h) * img_h * img_w

            r = np.random.uniform(r_1, r_2)

            w = int(np.sqrt(s / r))

            h = int(np.sqrt(s * r))

            left = np.random.randint(0, img_w)

            top = np.random.randint(0, img_h)



            if left + w <= img_w and top + h <= img_h:

                break



        if pixel_level:

            c = np.random.uniform(v_l, v_h, (h, w, img_c))

        else:

            c = np.random.uniform(v_l, v_h)



        input_img[top:top + h, left:left + w, :] = c



        return input_img



    return eraser
datagen = ImageDataGenerator(

    width_shift_range=0.5,

    height_shift_range=0.5,

    #horizontal_flip=True,

    #vertical_flip=True

    dtype=float,

    fill_mode='nearest',

    preprocessing_function = get_random_eraser(p=0.8, s_l=0.0009765625, s_h=0.0009765625, r_1=0.03124, r_2=0.03124, v_l=0, v_h=9, pixel_level=True),

)



x_train = x_train.reshape(x_train.shape + (1,) )

datagen.fit(x_train)



y_train = y_train.reshape(y_train.shape + (1,))



x_test = x_test.reshape(x_test.shape + (1,))

datagen.fit(x_test)
def sampling(args):

    z_mean, z_log_var = args

    batch = K.shape(z_mean)[0]

    dim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean = 0 and std = 1.0

    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon
original_dim = 32 * 32

input_shape_vae = (original_dim, )
x_vae_train = np.reshape(x_train, [-1, original_dim])

x_vae_test = np.reshape(x_test, [-1, original_dim])
x_vae_train.shape
intermediate_dim = 512

batch_size = 32

latent_dim = 256

epochs = 100
#def create_vae_model(batch_size, input_shape):



inputs = Input(shape=input_shape_vae)

x = Dense(intermediate_dim, activation='relu')(inputs)

x = Dropout(0.5)(x)

x = Dense(intermediate_dim, activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(intermediate_dim, activation='relu')(x)



#x = LSTM(512)(x)



z_mean = Dense(latent_dim)(x)

z_log_var = Dense(latent_dim)(x)



z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])



# instantiate encoder model

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

encoder.summary()



# build decoder model

latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')

x = Dense(intermediate_dim, activation='relu')(latent_inputs)

#x = LSTM(512, return_sequences=True)(x)

x = Dropout(0.3)(x)

x = Dense(intermediate_dim, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(intermediate_dim, activation='relu')(x)

outputs = Dense(original_dim, activation='sigmoid')(x)



# instantiate decoder model

decoder = Model(latent_inputs, outputs, name='decoder')

decoder.summary()



# instantiate VAE model

outputs = decoder(encoder(inputs)[2])

vae = Model(inputs, outputs, name='vae_mlp')
from keras.losses import mse, binary_crossentropy



#reconstruction_loss = mse(inputs, outputs)

reconstruction_loss = binary_crossentropy(inputs, outputs)



reconstruction_loss *= original_dim

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)

kl_loss = K.sum(kl_loss, axis=-1)

kl_loss *= -0.3

vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)



nadam = Nadam(learning_rate=0.003, beta_1=0.999, beta_2=0.999999)

vae.compile(optimizer=nadam)



vae.fit(x_vae_train,

        epochs=epochs,

        batch_size=batch_size,

        shuffle=True)
decoded_train = vae.predict(x_vae_train)



decoded_train = np.reshape(decoded_train, [-1, 32, 32, 1])
_max = np.amax(decoded_train)

_min = np.amin(decoded_train)



_range = _max - _min

_step = _range / 10



#decoded_train = (decoded_train - _min) / _range



decoded_train = (decoded_train * 18)

decoded_train = decoded_train.astype(int)
rd_train = []

for i in range(len(decoded_train)):

    w = ox_train[i].shape[0]

    h = ox_train[i].shape[1]

    if (decoded_train[i].shape[0] != h) | (decoded_train[i].shape[1] != w) :

        rd_train.append( cv2.resize(decoded_train[i], dsize=(h, w), interpolation=cv2.INTER_NEAREST) )
n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n+1):

    # 입력 출력

    ax = plt.subplot(3, n, i)

    plt.imshow(ox_train[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # 정답 출력

    ax = plt.subplot(3, n, i + n)

    plt.imshow(oy_train[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # 생성 출력

    ax = plt.subplot(3, n, i + 2 * n)

    plt.imshow(rd_train[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
decoded_test = vae.predict(x_vae_test)



decoded_test = np.reshape(decoded_test, [-1, 32, 32, 1])



_max = np.amax(decoded_test)

_min = np.amin(decoded_test)



_range = _max - _min

_step = _range / 10



#decoded_test = (decoded_test - _min) / _range



decoded_test = (decoded_test * 19)

decoded_test = decoded_test.astype(int)



print( np.amax(decoded_test) )

print( np.amin(decoded_test) )



rd_test = []

for i in range(len(decoded_test)):

    w = ox_test[i].shape[0]

    h = ox_test[i].shape[1]

    if (decoded_test[i].shape[0] != h) | (decoded_test[i].shape[1] != w) :

        rd_test.append( cv2.resize(decoded_test[i], dsize=(h, w), interpolation=cv2.INTER_NEAREST) )
n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n+1):

    # 입력 출력

    ax = plt.subplot(2, n, i)

    plt.imshow(ox_test[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # 생성 출력

    ax = plt.subplot(2, n, i + n)

    plt.imshow(rd_test[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
df_submission = pd.read_csv("../input/abstraction-and-reasoning-challenge/sample_submission.csv")

df_submission.head()
df_submission['output'][1]
'|' + str(rd_test[0]).replace('[','').replace(']','').replace('\n','|').replace(' ','') + '|'
for i, row in df_submission.iterrows():

    cand_0 = ''

    cand_1 = ''

    cand_2 = '|' + str(rd_test[i]).replace('[','').replace(']','').replace('\n','|').replace(' ','') + '|'

    answer = ''

    for j, cand in enumerate(row[1].split(' ')):

        #print(j)

        #print('-', cand)

        #print('=', cand_2)

        if j == 0: #cand_0

            cand_0 = cand

            answer += cand_0

            #print(answer)

        elif j == 1: #cand_1

            nums = []

            for c in cand_0.replace('|', ''):

                nums.append(int(c))

            for k, c in enumerate(cand_0):

                #print(k, c, cand_0)

                if c == '|':

                    cand_1 += '|'

                elif int(c) == np.amax(nums):

                    if np.amax(rd_test[i]) == cand_2[k]:

                        cand_1 += c

                    else:

                        cand_1 += cand_2[k]

                else:

                    cand_1 += c

            answer += ' ' + cand_1

            #print('+2', answer)

        elif j == 2: #cand_2

            answer += ' ' + cand_2

            #print(answer)

    #print(answer)

    #print()

    df_submission.at[i,'output'] = answer
df_submission.head()
df_submission.to_csv("submission.csv", index=False)