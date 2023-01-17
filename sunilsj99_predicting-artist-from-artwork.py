# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import glob

import cv2

import os

%matplotlib inline

# Any results you write to the current directory are saved as output.
folders = glob.glob('../input/images/images/*')



Images = []

Labels = []

for folder in folders:

    for f in glob.glob(folder+'/*.jpg'):

        Images.append(f)

        Labels.append(folder)
data = {'Artwork': Images, 'Artist': Labels}



df = pd.DataFrame(data)



df.head()
artists = []

for i in range(df.shape[0]):

    a = df['Artist'].iloc[i]

    artists.append(a.split('/')[-1])
df['Artist'] = artists
df.head(10)
df = df.sample(frac=1).reset_index(drop = True)



df.head()
img0 = cv2.imread(df['Artwork'].iloc[0])



plt.imshow(img0)

print(str(img0.shape)+" Artist : "+str(df['Artist'].iloc[0]))
plt.figure(figsize=(15,10))

sns.countplot(y=df['Artist'])
from keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(rescale=1./255., zoom_range=0.2, rotation_range=0.2, horizontal_flip=True, vertical_flip=True,

                               shear_range=0.2, validation_split=0.25)
train_generator = train_gen.flow_from_dataframe(df, directory='', x_col='Artwork', y_col='Artist', batch_size=32,

                                                subset="training", seed=42, target_size=(299,299), shuffle=True)



valid_generator = train_gen.flow_from_dataframe(df, directory='', x_col='Artwork', y_col='Artist', batch_size=32,

                                                subset="validation", seed=42, target_size=(299,299), shuffle=True)
from keras import applications

from keras import optimizers, regularizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Activation

from keras import backend as k 

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint
model = applications.xception.Xception(weights = 'imagenet', include_top = False, input_shape = (299,299,3))
x = model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

predictions = Dense(50, activation="softmax")(x)
model_final = Model(inputs = model.input, outputs = predictions)
model_final.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001), metrics=["accuracy"])
history = model_final.fit_generator(train_generator, 

                                   steps_per_epoch = train_generator.n//train_generator.batch_size,

                                    validation_data= valid_generator,

                                    validation_steps= valid_generator.n//valid_generator.batch_size

                                   ,epochs = 20

                                   ,callbacks=[es, mc])
from keras.models import load_model, save_model



final_model = load_model('best_model.h5')
final_model.save('final_model.h5')