import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import os

from tqdm import tqdm



import cv2
train = pd.read_csv("../input/shopee-code-league-2020-product-detection/train.csv")

print(train.shape)

train.head(10)
plt.figure(figsize=(10,10))

plt.title("Distribution of labels for training data")

sns.countplot(train['category'])
new_train=pd.DataFrame()



CATEGORIES=[n for n in range(10)]



for cat in CATEGORIES:

    new_train=new_train.append(train[train['category']==cat][:1600])



del train



train=new_train.sample(frac=1)

train
plt.figure(figsize=(10,10))

sns.countplot(train['category'])
resized_img_dim=150



def read_img(train,resized_img_dim):

    

    DATADIR='../input/shopee-code-league-2020-product-detection/resized/train'

    X=[]



    for fname,cat in tqdm(train.values):

        if(cat<10):

            cat='0'+str(cat)

        else:

            cat=str(cat)



        path=os.path.join(DATADIR,cat,fname)



        try:

            img=cv2.imread(path).astype('float32')

            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img=cv2.resize(img, (resized_img_dim,resized_img_dim))

        except:

            pass

        X.append(img)

    return X



X=read_img(train,resized_img_dim)
from sklearn.preprocessing import OneHotEncoder



y=train['category']



ohe=OneHotEncoder()

y=ohe.fit_transform(y.values.reshape(-1,1)).astype('float32')

y=y.todense()
from sklearn.model_selection import train_test_split



X=np.array(X).reshape(-1,resized_img_dim,resized_img_dim,3)



xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)
del X

del y

del train
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=30,

                                   width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, 

                                   horizontal_flip=True, fill_mode='constant')



val_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow(xtrain, ytrain, batch_size=30)

val_generator = val_datagen.flow(xtest, ytest, batch_size=20)
plt.figure(figsize=(10,10))

for xbatch,ybatch in train_generator:

    for i in range(1,10):

        plt.subplot(3,3,i)

        plt.axis('off')

        plt.imshow(((xbatch[i]*255).astype('uint8')))

    break
import tensorflow as tf

import keras

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization

from keras.applications.vgg19 import VGG19



input_shape_=(resized_img_dim,resized_img_dim,3)



vgg=VGG19(include_top=False, input_shape=input_shape_)



output = vgg.layers[-1].output

output = Flatten()(output)



vgg_model=Model(vgg.input,output)





print(vgg_model.summary())
pretrained_model = Sequential()



pretrained_model.add(vgg_model)



pretrained_model.add(Dense(256,activation='relu', input_dim=input_shape_))

pretrained_model.add(Dropout(0.4))



pretrained_model.add(Dense(10, activation='softmax'))



pretrained_model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.RMSprop(lr=2e-5),

              metrics=['accuracy'])



model_callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)]



history = pretrained_model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,

                              validation_data=val_generator, validation_steps=50, 

                              verbose=1, callbacks=model_callbacks)  
#Accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Training accuracy vs Validation accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.show()



#Loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Training loss vs Validation loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.show()
score=pretrained_model.evaluate(xtest,ytest)