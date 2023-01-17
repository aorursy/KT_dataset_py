# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/emnist/emnist-balanced-train.csv', sep=',',header=None)  ##reading the training csv it has 784 columns representing 28*28 pixels in the image 

df_test = pd.read_csv('../input/emnist/emnist-balanced-test.csv', sep=',', header=None)   ##reading the testing csv
df_train.shape
df_train.head()
import matplotlib.pyplot as plt

def show_images(images):

    """

    images : numpy arrays

    """

    n_images = len(images)

    titles = ['(%d)' % i for i in range(1, n_images + 1)]

    num = 5

    iter_num = np.ceil(n_images / num).astype(int)

    for i in range(iter_num):

        fig = plt.figure()

        sub_images = images[i * num:num * (i + 1)]

        sub_titles = titles[i * num:num * (i + 1)]

        for n, (image, title) in enumerate(zip(sub_images, sub_titles)):

            a = fig.add_subplot(1, np.ceil(len(sub_images)), n + 1)

            if image.ndim == 2:

                plt.gray()

            a.set_title(title, fontsize=15)

            plt.imshow(image)

            

            

for i in range(0,47):

    p = df_train[df_train[0]==i].head(10)

    p = p.drop(0, axis=1)

    p = p.values

    p = p.reshape(p.shape[0],28,28)

    show_images(p)

    

    

#as you can see the images are rotated by 90
characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']

print(len(characters))
x_train = df_train.drop(0,axis=1).values

x_test = df_test.drop(0, axis=1).values

y_train = df_train[0].values

y_test = df_test[0].values
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
x_train = x_train.reshape(x_train.shape[0],28,28)

x_test = x_test.reshape(x_test.shape[0],28,28)

print(x_train.shape)

print(x_test.shape)
x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255

temp = x_train[:10]

show_images(x_train[:10])
for i in range(len(temp)):

    temp[i] = np.flip(temp[i],1)

    temp[i] = np.rot90(temp[i])

    

show_images(temp)

def show_images(images):

    """

    images : numpy arrays

    """

    n_images = len(images)

    for i in range(len(images)):

        images[i] = np.flip(images[i],1)

        images[i] = np.rot90(images[i])

        

        

    titles = ['(%d)' % i for i in range(1, n_images + 1)]

    num = 5

    iter_num = np.ceil(n_images / num).astype(int)

    for i in range(iter_num):

        fig = plt.figure()

        sub_images = images[i * num:num * (i + 1)]

        sub_titles = titles[i * num:num * (i + 1)]

        for n, (image, title) in enumerate(zip(sub_images, sub_titles)):

            a = fig.add_subplot(1, np.ceil(len(sub_images)), n + 1)

            if image.ndim == 2:

                plt.gray()

            a.set_title(title, fontsize=15)

            plt.imshow(image)

            

            

for i in range(0,47):

    p = df_train[df_train[0]==i].head(10)

    p = p.drop(0, axis=1)

    p = p.values

    p = p.reshape(p.shape[0],28,28)

    show_images(p)
for i in range(len(x_train)):

    x_train[i] = np.flip(x_train[i],1)

    x_train[i] = np.rot90(x_train[i])

    

for i in range(len(x_test)):

    x_test[i] = np.flip(x_test[i],1)

    x_test[i] = np.rot90(x_test[i])
temp = x_train[:10]

show_images(x_train[:10])
from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=47, dtype='float32')

y_test = to_categorical(y_test, num_classes=47, dtype='float32')
print(y_train.shape)

print(y_test.shape)

print(y_train[0])
from skimage.transform import resize



target_size = 96



def preprocess_image(x):

    # Resize the image to have the shape of (96,96)

    x = resize(x, (target_size, target_size),

            mode='constant',

            anti_aliasing=False)

    

    # convert to 3 channel (RGB)

    x = np.stack((x,)*3, axis=-1) 

    #print(x.shape)

    return x.astype(np.float32)
from sklearn.utils import shuffle

def load_data_generator(x, y, batch_size=64):

    num_samples = x.shape[0]

    while 1:  # Loop forever so the generator never terminates

        try:

            shuffle(x)

            for i in range(0, num_samples, batch_size):

                x_data = [preprocess_image(im) for im in x[i:i+batch_size]]

                y_data = y[i:i + batch_size]

            

                # convert to numpy array since this what keras required

                yield shuffle(np.array(x_data), np.array(y_data))

        except Exception as err:

            print(err)
from keras.applications.mobilenetv2 import MobileNetV2

from keras.layers import Dense, Input, Dropout

from keras.models import Model



def build_model( ):

    input_tensor = Input(shape=(target_size, target_size, 3))

    base_model = MobileNetV2(

        include_top=False,

        weights='imagenet',

        input_tensor=input_tensor,

        input_shape=(target_size, target_size, 3),

        pooling='avg')



    for layer in base_model.layers:

        layer.trainable = True  # trainable has to be false in order to freeze the layers

        

    op = Dense(1024, activation='relu')(base_model.output)

    op = Dropout(.25)(op)

    

    

    ##

    # softmax: calculates a probability for every possible class.

    #

    # activation='softmax': return the highest probability;

    # for example, if 'Coat' is the highest probability then the result would be 

    # something like [0,0,0,0,1,0,0,0,0,0,...] with 1 in index 5 indicate 'Coat' in our case.

    ##

    output_tensor = Dense(47, activation='softmax')(op)



    model = Model(inputs=input_tensor, outputs=output_tensor)





    return model
from keras.optimizers import Adam

model = build_model()

model.compile(optimizer=Adam(),

              loss='categorical_crossentropy',

              metrics=['categorical_accuracy'])

model.summary()
train_generator = load_data_generator(x_train, y_train, batch_size=64)

test_generator = load_data_generator(x_test, y_test, batch_size=64)



model.fit_generator(generator=train_generator,steps_per_epoch=5,verbose=1,epochs=50)

    

score = model.evaluate_generator(generator=test_generator,steps=900,verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
##In there research paper they got a baseline accuracy of 70% on this model



model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
from scipy.misc import imread, imresize,imshow

x = imread('../input/emnist-sample-image/output.png',mode='L')

x = np.invert(x)

x = np.flip(x,1)

x = np.rot90(x)

x = preprocess_image(x)

show_images([x])

print(x.shape)
x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])

print(x.shape)
out = model.predict(x)

print(out)

p = np.argmax(out,axis=1)

print(characters[p[0]-1])