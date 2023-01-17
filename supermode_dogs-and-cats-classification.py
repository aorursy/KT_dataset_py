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
import cv2



import matplotlib.pyplot as plt

%matplotlib inline 



#To see our directory

import random

import gc   #Gabage collector for cleaning deleted data from memory
train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train'

test_dir = '../input/dogs-vs-cats-redux-kernels-edition/test'



train_dogs = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]  #get dog images

train_cats = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  #get cat images



test_imgs = ['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images



train_imgs = train_dogs[:] + train_cats[:]  # slice the dataset and use 2000 in each

random.shuffle(train_imgs)  # shuffle it randomly



#Clear list that are useless

del train_dogs

del train_cats

gc.collect()   #collect garbage to save memory
nrows = 150

ncolumns = 150

channels = 3  #change to 1 if you want to use grayscale image





#A function to read and process the images to an acceptable format for our model

def read_and_process_image(list_of_images):

    """

    Returns two arrays: 

        X is an array of resized images

        y is an array of labels

    """

    X = [] # images

    y = [] # labels

    

    for image in list_of_images:

        if 'jpg' in image:

            X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image

        #get the labels

        if 'dog' in image.split('/')[4]:

            y.append(1)

        elif 'cat' in image.split('/')[4]:

            y.append(0)

    

    return X, y

X, y = read_and_process_image(train_imgs)
plt.figure(figsize=(20,10))

columns = 5

for i in range(columns):

    plt.subplot(5 / columns + 1, columns, i + 1)

    plt.imshow(X[i])
import seaborn as sns

del train_imgs

gc.collect()



#Convert list to numpy array

X = np.array(X)

y = np.array(y)



#Lets plot the label to be sure we just have two class

sns.countplot(y)

plt.title('Labels for Cats and Dogs')
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)



print("Shape of train images is:", X_train.shape)

print("Shape of validation images is:", X_val.shape)

print("Shape of labels is:", y_train.shape)

print("Shape of labels is:", y_val.shape)
del X

del y

gc.collect()



#get the length of the train and validation data

ntrain = len(X_train)

nval = len(X_val)



#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***

batch_size = 32
from keras.applications import InceptionResNetV2

path = '../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

conv_base = InceptionResNetV2(weights= path, include_top=False, input_shape=(150,150,3))
from keras import layers

from keras import models



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes
model.summary()
print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
from keras import optimizers



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img



train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1

                                    rotation_range=40,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')



val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale
train_generator = train_datagen.flow(X_train, y_train,batch_size=batch_size)

val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)



del X_train

del X_val

del y_train

del y_val

gc.collect()
history = model.fit_generator(train_generator,

                              steps_per_epoch=ntrain // batch_size,

                              epochs=20,

                              validation_data=val_generator,

                              validation_steps=nval // batch_size)
model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
del val_generator

del train_generator

gc.collect()
X_test, y_test = read_and_process_image(test_imgs) #Y_test in this case will be empty.

X_test = np.array(X_test).astype('float32')

X_test /= 255

del test_imgs

gc.collect()

    

print("Test dataset shape: ")

print(X_test.shape)

predictions = model.predict(X_test, verbose=0)

del X_test

gc.collect()
test_imgs = ['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)] #get test images

X = []

for i in test_imgs:

    if '.jpg' in i:

        X.append(int(i.split('/')[4].replace('.jpg', '')))

solution = pd.DataFrame({"id": X, "label":list(predictions)})

cols = ['label']



for col in cols:

    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)



solution.to_csv("dogsVScats.csv", index = False)