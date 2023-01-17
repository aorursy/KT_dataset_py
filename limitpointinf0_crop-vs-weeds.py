import os

from PIL import Image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

import scipy.misc

from skimage import transform

import warnings



warnings.filterwarnings("ignore")



#convertering list of training data paths to df

train_dir = '../input/train/'

train_list = os.listdir(train_dir)

records = []

for category in train_list:

    img_list = os.listdir(train_dir + category)

    for img in img_list:

        records.append((img,category))

        

df_train = pd.DataFrame.from_records(records,columns=['image','category'])



print(df_train.head())







#looking at the test data

test_dir = '../input/test/'

test_list = os.listdir(test_dir)

print('Train Data', len(df_train.index))

print('Test Data',type(test_list),len(test_list))

print('categories',os.listdir(train_dir))

print('# of categories', len(os.listdir(train_dir)))
for i in list(df_train['image'])[0:1]:

    img = Image.open(train_dir + df_train['category'][0] + '/' + i)

    img.load()

    data = np.asarray(img, dtype="float32" )

    plt.imshow(data)

    plt.show()
dim_image = []

for i in (train_dir + df_train['category'] + '/' + df_train['image']):

    img = Image.open(i)

    data = img.size

    dim_image.append(data[0])

print('smallest image dimension', min(dim_image))



i_height = min(dim_image)

i_width = min(dim_image)



X = []

count = 0

bad_images = []

#df_train = df_train.drop(df_train.index[bad_images])

for i in (train_dir + df_train['category'] + '/' + df_train['image']):

    img = Image.open(i)

    img.load()

    img = np.asarray(img, dtype='float32')

    img = img/255

    data = transform.resize(img,(49,49))

    if data.size != 7203:

        bad_images.append(count)

#     plt.imshow(data)

#     plt.show()

#     X.append(data)

    count += 1

print('bad images',bad_images)



df_train = df_train.drop(df_train.index[bad_images])

for i in (train_dir + df_train['category'] + '/' + df_train['image']):

    img = Image.open(i)

    img.load()

    img = np.asarray(img, dtype='float32')

    img = img/255

    data = transform.resize(img,(49,49))

    X.append(data)



X = np.array(X)



y = np.array(df_train['category'].astype('category').cat.codes)



print('Done creating X and y.')

print('X Shape',X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



im_shape = (49,49,3)

batch_size = 10



cnn  = Sequential([

    Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=im_shape, padding='same'),

    LeakyReLU(alpha=0.1),

    MaxPooling2D((2,2), padding='same'),

    Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'),

    LeakyReLU(alpha=0.1),

    MaxPooling2D((2,2), padding='same'),

    Conv2D(128, kernel_size=(3,3), activation='linear', padding='same'),

    LeakyReLU(alpha=0.1),

    MaxPooling2D((2,2), padding='same'),

    Flatten(),

    Dense(50,activation='relu'),

    Dense(12, activation='softmax')

])



cnn.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])



cnn. fit(X_train, y_train, batch_size=batch_size, epochs=10, verbose=1, validation_data=(X_test,y_test))