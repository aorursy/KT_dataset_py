# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from PIL import Image

imgs = []

gray_imgs = []

gray_imgs3chan = []

one_hot_list = []



# Choose the model you want to run

modelnum = 2 # 1 for U-Net, 2 for U-Net with dense branch attached, 3 for U-Net with autoencoder attached



# Read in the csv file which contains pokemon types

df = pd.read_csv('/kaggle/input/pokemon-images-and-types/pokemon.csv')



# One hot encode the pokemon's primary type

one_hot = pd.get_dummies(df['Type1'])



# Go through add grab every image and save them as numpy arrays in a list

for dirname, _, filenames in os.walk('/kaggle/input/pokemon-images-and-types/images/'):

    for filename in filenames:

        # split the file name string to find what type of pokemon the image contains

        pokemonname = filename.split('/')[-1].split('.')[0]

        rownum = df.loc[df['Name'] == pokemonname].index[0]

        encoded =np.array(one_hot.iloc[rownum,:]).astype(np.float32)

        one_hot_list.append(encoded)

        

        # Open the image with PIL and save it as grayscale color and grayscale but in RGB format

        img = Image.open(os.path.join(dirname, filename)).convert('RGB') # Some are RGBA so we convert them

        gray_img = img.convert('L') # Save a gray scale version too

        imgs.append(np.array(img))

        gray_imgs.append(np.array(gray_img))

        gray_imgs3chan.append(np.array(gray_img.convert('RGB'))) # This is a 3 channel version of the gray scale image.



# Convert lists to numpy arrays for tensorflow processing

one_hot_list = np.array(one_hot_list)

imgs = np.array(imgs)

gray_imgs = np.array(gray_imgs)

gray_imgs = np.expand_dims(gray_imgs, axis=-1)

gray_imgs3chan = np.array(gray_imgs3chan)

print(imgs.shape, gray_imgs.shape, one_hot_list.shape)





# Any results you write to the current directory are saved as output.
# I just used these commands for data exploration. I haven't used much Pandas before

df = pd.read_csv('/kaggle/input/pokemon-images-and-types/pokemon.csv')

df.head()

df.Type2.value_counts()

df.Type2.isna().sum()

#len(df.Type1.value_counts()) # there are 18 different pokemon types

df.head()





df.loc[df['Name'] == 'charizard'].Type1.iloc[0] == 'Fire'

#df['Name'].where(df['Name'] == 'charizard')

one_hot = pd.get_dummies(df['Type1'])

np.array(one_hot.iloc[2,:]).astype(np.float32)

print("ignore this")
f,ax = plt.subplots(10,2) 

f.subplots_adjust(0,0,3,3)

for i in range(0,10,1):

    ax[i,1].imshow(gray_imgs[i,:,:,0], cmap=plt.get_cmap('gray'))

    ax[i,0].imshow(imgs[i,:,:])
# break the data into a training and testing partition





testdatasize = 30

train_gray = gray_imgs[:-testdatasize]/255

test_gray = gray_imgs[-testdatasize:]/255

train_color = imgs[:-testdatasize]/255

test_color = imgs[-testdatasize:]/255

gray_imgs3chan = gray_imgs3chan[:-testdatasize]/255 # I can use this for pretraining

train_oh = one_hot_list[:-testdatasize]

test_oh = one_hot_list[-testdatasize:]



print(train_gray.shape, train_color.shape, train_oh.shape)
# data augmentation by rotation (not currently using this)

def D4aug(arr):

    r1 = np.rot90(arr,k=1,axes=(1,2))

    r2 = np.rot90(arr,k=2,axes=(1,2))

    r3 = np.rot90(arr,k=3,axes=(1,2))

    return np.concatenate((arr,r1,r2,r3),axis=0)



#train_gray = D4aug(train_gray)

#train_color = D4aug(train_color)

#gray_imgs3chan = D4aug(gray_imgs3chan)

#print(train_gray.shape,gray_imgs3chan.shape)
# import tensorflow 

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose, SpatialDropout2D, Dense, Add, Flatten
# Define the something like the U-Net Model

def unet():

    X = tf.keras.Input(shape=(120,120,1))

    l1 = Conv2D(64, (3,3), padding='same', activation='relu')(X)

    l2 = Conv2D(64, (3,3), padding='same', activation='relu')(l1)

    

    MP1 = MaxPooling2D((2,2),strides=(2,2))(l2)

    MP1 = SpatialDropout2D(.1)(MP1)

    

    l3 = Conv2D(128, (3,3), padding='same', activation='relu')(MP1)

    l4 = Conv2D(128, (3,3), padding='same', activation='relu')(l3)

    MP2 = MaxPooling2D((2,2),strides=(2,2))(l4)

    MP2 = SpatialDropout2D(.2)(MP2)

    

    l5 = Conv2D(128, (3,3), padding='same', activation='relu')(MP2)

    l6 = Conv2D(128, (3,3), padding='same', activation='relu')(l5)

    MP3 = MaxPooling2D((2,2),strides=(2,2))(l6)

    MP3 = SpatialDropout2D(.2)(MP3)

    

    bn1 = Conv2D(256, (3,3), padding='same', activation='relu')(MP3)

    bn2 = Conv2D(256, (3,3), padding='same', activation='relu')(bn1)

    bn2 = SpatialDropout2D(.2)(bn2)

    

    

    

    u1 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(bn2)

    conc1 = Concatenate()([u1,l6])

    c1 = Conv2D(3,(3,3),padding='same', activation='relu')(conc1)

    

    u2 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(conc1)

    conc2 = Concatenate()([u2,l4])

    c2 = Conv2D(3,(3,3),padding='same', activation='relu')(conc2)

    

    u3 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(conc2)

    conc3 = Concatenate()([u3,l2])

    #conc3 = SpatialDropout2D(.15)(conc3)

    c3 = Conv2D(3,(3,3),padding='same', activation='sigmoid')(conc3)

    

    model = tf.keras.Model(X,c3)

    return model
# You can use this callback if you want

# if you don't let the model overfit then the pokemon are not very colorful, which is sad

es = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto',

    baseline=None, restore_best_weights=True)

# Below is code to pretrain. You are simply training the model to output the same grayscale image

# I did not observe any benefit from pretraining, which is what you would expect when using ReLU activations

# and skip connection. It is easy to learn the identity map

# I tried pretraining because the dataset is so small

# Un-comment the next line to pretrain

#model.fit(train_gray,gray_imgs3chan,40,20,validation_split=.05,callbacks=[es])
def unet_with_type():

    

    type_in = tf.keras.Input(shape=(18,))

    d1 = Dense(2**9, activation='relu')(type_in)

    d2 = Dense(2**8, activation='relu')(d1)

    

    d2 = Dense(15*15,activation='relu')(d2)

    d2 = tf.keras.layers.Reshape((15,15,1))(d2)

    

    

    # Below is the unet

    X = tf.keras.Input(shape=(120,120,1))

    l1 = Conv2D(64, (3,3), padding='same', activation='relu')(X)

    l2 = Conv2D(64, (3,3), padding='same', activation='relu')(l1)

    

    MP1 = MaxPooling2D((2,2),strides=(2,2))(l2)

    MP1 = SpatialDropout2D(.1)(MP1)

    

    l3 = Conv2D(128, (3,3), padding='same', activation='relu')(MP1)

    l4 = Conv2D(128, (3,3), padding='same', activation='relu')(l3)

    MP2 = MaxPooling2D((2,2),strides=(2,2))(l4)

    MP2 = SpatialDropout2D(.2)(MP2)

    

    l5 = Conv2D(128, (3,3), padding='same', activation='relu')(MP2)

    l6 = Conv2D(128, (3,3), padding='same', activation='relu')(l5)

    MP3 = MaxPooling2D((2,2),strides=(2,2))(l6)

    MP3 = SpatialDropout2D(.2)(MP3)

    

    bn1 = Conv2D(256, (3,3), padding='same', activation='relu')(MP3)

    bn1 = Add()([bn1,d2])

    bn2 = Conv2D(256, (3,3), padding='same', activation='relu')(bn1)

    bn2 = SpatialDropout2D(.2)(bn2)

    

    

    

    u1 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(bn2)

    conc1 = Concatenate()([u1,l6])

    c1 = Conv2D(3,(3,3),padding='same', activation='relu')(conc1)

    

    u2 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(conc1)

    conc2 = Concatenate()([u2,l4])

    c2 = Conv2D(3,(3,3),padding='same', activation='relu')(conc2)

    

    u3 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(conc2)

    conc3 = Concatenate()([u3,l2])

    #conc3 = SpatialDropout2D(.15)(conc3)

    c3 = Conv2D(3,(3,3),padding='same', activation='sigmoid')(conc3)

    

    model = tf.keras.Model([X,type_in],c3)

    return model



def auto_with_unet():

    type_in = tf.keras.Input(shape=(18,))

    d1 = Dense(2**9, activation='relu')(type_in)

    d2 = Dense(2**8, activation='relu')(d1)

    

    d2 = Dense(15*15,activation='relu')(d2)

    d2 = tf.keras.layers.Reshape((15,15,1))(d2)

    

    

    # Below is the unet

    X = tf.keras.Input(shape=(120,120,1))

    l1 = Conv2D(64, (3,3), padding='same', activation='relu')(X)

    l2 = Conv2D(64, (3,3), padding='same', activation='relu')(l1)

    

    MP1 = MaxPooling2D((2,2),strides=(2,2))(l2)

    MP1 = SpatialDropout2D(.1)(MP1)

    

    l3 = Conv2D(128, (3,3), padding='same', activation='relu')(MP1)

    l4 = Conv2D(128, (3,3), padding='same', activation='relu')(l3)

    MP2 = MaxPooling2D((2,2),strides=(2,2))(l4)

    MP2 = SpatialDropout2D(.2)(MP2)

    

    l5 = Conv2D(128, (3,3), padding='same', activation='relu')(MP2)

    l6 = Conv2D(128, (3,3), padding='same', activation='relu')(l5)

    MP3 = MaxPooling2D((2,2),strides=(2,2))(l6)

    MP3 = SpatialDropout2D(.2)(MP3)

    

    bn1 = Conv2D(256, (3,3), padding='same', activation='relu')(MP3)

    bn1 = Add()([bn1,d2])

    bn2 = Conv2D(256, (3,3), padding='same', activation='relu')(bn1)

    bn2 = SpatialDropout2D(.2)(bn2)

    

    

    

    u1 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(bn2)

    conc1 = Concatenate()([u1,l6])

    c1 = Conv2D(3,(3,3),padding='same', activation='relu')(conc1)

    

    u2 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(conc1)

    conc2 = Concatenate()([u2,l4])

    c2 = Conv2D(3,(3,3),padding='same', activation='relu')(conc2)

    

    u3 = Conv2DTranspose(64,(3,3),strides=(2,2), padding='same', activation='relu')(conc2)

    conc3 = Concatenate()([u3,l2])

    #conc3 = SpatialDropout2D(.15)(conc3)

    c3 = Conv2D(3,(3,3),padding='same', activation='sigmoid')(conc3)

    

    # output of autoencoder

    e0 = Flatten()(bn2)

    e1 = Dense(15*15,activation='relu')(e0)

    e2 = Dense(2**8,activation='relu')(e1)

    e3 = Dense(2**9,activation='relu')(e2)

    eout = Dense(18,activation='sigmoid')(e3)

    model = tf.keras.Model([X,type_in],[c3,eout])

    return model



myadam = tf.keras.optimizers.Adam(learning_rate=0.001/3.0, beta_1=0.9, beta_2=0.999, amsgrad=False)

if modelnum == 1:

    model = unet()

    model.compile('adam', loss='MSE') # for training unet and auto_with_unet

elif modelnum == 2:

    model = unet_with_type()

    model.compile('adam', loss='MSE') # for training unet and auto_with_unet

else:

    model = auto_with_unet()

    model.compile(myadam,loss=['MSE','categorical_crossentropy'])

model.summary()
# train with types

if modelnum == 2:

    model.fit([train_gray,train_oh],train_color,10,150,validation_split=.05,callbacks=None)

# train unet

elif modelnum == 1:

    model.fit(train_gray, train_color,10,150,validation_split=.05,callbacks=None)

else:

    model.fit([train_gray,train_oh],[train_color,train_oh],10,150,validation_split=.05,callbacks=None)
# examine results when trained with types

if modelnum == 2:

    print('MSE: ',model.evaluate([test_gray,test_oh],test_color))

    preds = model.predict([test_gray,test_oh])



# examine results when trained w/o types

elif modelnum == 1:

    print("MSE loss: ", model.evaluate(test_gray,test_color))

    preds = model.predict(test_gray)



# examine results of auto_with_unet

else:

    print(model.evaluate([test_gray,test_oh],[test_color,test_oh]))

    preds, garb = model.predict([test_gray,test_oh])

numimgs = 10

f,ax = plt.subplots(numimgs,2,figsize=[6.4*2,4.8*2]) 

f.subplots_adjust(0,0,2,2)

for i in range(0,numimgs,1):

    ax[i,0].imshow(Image.fromarray( (preds[i]*255).astype(np.uint8)))

    ax[i,1].imshow(Image.fromarray( (test_color[i]*255).astype(np.uint8)))

    

x = [axi.set_axis_off() for axi in ax.ravel()]