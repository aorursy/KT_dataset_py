# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import style
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Input,MaxPooling2D,Dropout
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
style.use('ggplot')
%matplotlib notebook
# Any results you write to the current directory are saved as output.
base_dir = '../input/dataset/dataset'

#train autoencoder files
train_dir = os.path.join(base_dir,'training_set')

#train classification model files
test_dir = os.path.join(base_dir,'test_set')

train_cats = os.path.join(train_dir,'cats')
train_dogs = os.path.join(train_dir,'dogs')

test_cats = os.path.join(test_dir,'cats')
test_dogs = os.path.join(test_dir,'dogs')

#get filenames
train_cats_fnames = os.listdir(train_cats)
train_dogs_fnames = os.listdir(train_dogs)

#visualize sample images

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

pic = [os.path.join(train_cats, train_cats_fnames[0])]
pic2 = [os.path.join(train_dogs, train_dogs_fnames[0])]

for i, img in enumerate(pic + pic2):
    ax = plt.subplot(1,2,i+1)
    image = mpimg.imread(img)
    ax.axis('off')
    plt.imshow(image)
    print(image.shape)

input_img = Input(shape=(152,152, 3)) #RGB image

#encoder
x = Conv2D(16,(3,3),activation='relu',padding='same',)(input_img)
x = MaxPooling2D(2,padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D(2,padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoded = MaxPooling2D(2,padding='same')(x)

# at this point the representation is (19, 19, 8)

#decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder=Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()
# use all files in 'training' directory data to train autoencoder
# augment train data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1./255,
                          rotation_range=40, 
                          width_shift_range=0.2,
                          height_shift_range=0.2, 
                          shear_range=.2)

train_gen = train.flow_from_directory(train_dir,
                                    target_size=(152,152),
                                    batch_size=40,
                                    class_mode=None,
                                   )
#use "test" directory to train classification model

test = ImageDataGenerator(rescale=1./255, 
                          validation_split=.4) # split data to train/validation subsets

train_generator = test.flow_from_directory(test_dir,
                                  target_size=(152,152),
                                  batch_size=10,
                                  class_mode='binary',
                                  subset='training'
)
test_generator = test.flow_from_directory(test_dir,
                                  target_size=(152,152),
                                  batch_size=10,
                                  class_mode='binary',
                                  subset='validation'
)
#fit autoencoder model with training data

 
def fixed_generator(generator): # autoencoder don't feed labels, but 'fit_generator' need x,y values
    for batch in generator:
        yield (batch, batch)

hist = autoencoder.fit_generator(
    fixed_generator(train_gen),
    epochs = 10,
    steps_per_epoch=200)
#compare original images with reconstructions
%matplotlib inline
n = 10
x = train_gen.next()
decoded_imgs = autoencoder.predict(x)


plt.figure(figsize=(20, 4))
for i in range(n):
#     # display original
    ax = plt.subplot(2, n, i+1)
    image = x[i]
    plt.imshow(image.reshape(152, 152, 3)) #RGB image
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(decoded_imgs[i].reshape(152, 152, 3))#RGB image
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#visualize activations

try:
    layer_outputs = [layer.output for layer in autoencoder.layers[1:2]]
    activation_model = Model(inputs=autoencoder.input, outputs=layer_outputs)
    activations = activation_model.predict(x)
except:
    pass #avoid any errors

n = activations.shape[-1]

fig=plt.figure(figsize=(8, 8))
rows = 4
columns = n//rows
for i in range(n):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(activations[0, :, :, i])
    plt.axis('off')
#encoder 
input_img = Input(shape=(152,152, 3)) #RGB image

#same encoder part
x = Conv2D(16,(3,3),activation='relu',padding='same',)(input_img)
x = MaxPooling2D(2,padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D(2,padding='same')(x)
x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoded = MaxPooling2D(2,padding='same')(x)

# fully connected layer with output
model = Flatten()(encoded)
model = Dense(1024, activation='elu')(model)
model = keras.layers.BatchNormalization()(model)
model = Dropout(0.5)(model)
model = Dense(1,activation='sigmoid')(model) #output

full_model = Model(input_img, model)
# copy encoder-model weights 

for l1,l2 in zip(full_model.layers[:7],autoencoder.layers[0:7]):
    l1.set_weights(l2.get_weights())
#froze encoder layers

for layer in full_model.layers[0:7]:
    layer.trainable = False
full_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
full_model.summary()
callbacks =keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=20, 
                                        )
hist = full_model.fit_generator(train_generator,
                                     epochs=100,
                                     steps_per_epoch=100,
                                     validation_data=test_generator,
                                     validation_steps=100,
                                     callbacks=[callbacks])
from matplotlib import style
style.use('dark_background')
%matplotlib inline
def show_results(hists):
    for i,hist in enumerate(hists):
        plt.figure(figsize=(12,6))
        plt.subplot(len(hists),2,1)
        acc = hist.history['acc']
        test_acc = hist.history['val_acc']
        loss=hist.history['loss']
        test_loss=hist.history['val_loss']
        epochs=range(len(acc))
        plt.plot(epochs,test_acc,label='test')
        plt.plot(epochs,acc,label='train')
        plt.axhline(y=max(test_acc), linestyle='--')
        plt.legend()
        plt.title('Accuracy_{}'.format(i))
        plt.subplot(len(hists),2,2)
        plt.plot(epochs,test_loss,label='test')
        plt.plot(epochs,loss,label='train')
        plt.axhline(y=min(test_loss),linestyle='--')
        plt.legend()
        plt.title('Loss_{}'.format(i))
show_results([hist])
