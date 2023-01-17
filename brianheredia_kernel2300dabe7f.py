# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import matplotlib.pyplot as plt

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/training/training.csv')
df.head()
print(df.isnull().any().value_counts(), df.shape)

df.dropna(inplace=True)

#df.fillna(method = 'ffill',inplace = True)

#df.reset_index(drop = True, inplace = True)

print(df.isnull().any().value_counts(), df.shape)
df = df.sample(frac=1)

img_data = df['Image'].values

df.drop('Image', inplace=True, axis=1)
images = [np.array(list(map(int, image.split())), dtype=np.uint8).reshape(96,96) for image in img_data]

df.head()
# example images

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,20))

c = 0

r = 0

for i in range(9):

    ax[r, c].imshow(images[i], 'gray')

    for j in range(0, len(list(df)), 2):

        ax[r,c].scatter(df.loc[i][j], df.loc[i][j+1], s=15, c='red')

    ax[r,c].axis('off')

    c += 1

    if c == 3:

        r += 1

        c = 0
import tensorflow as tf

from tensorflow import keras



def build_model():

    # output has to be 30 values: 15 x coordinates and 15 y coordinates corresponding to the facial keypoints

    inp = keras.layers.Input(shape=(96,96,1))

    x = keras.layers.Conv2D(16, (2,2), activation='relu', padding='same')(inp)

    x = keras.layers.MaxPool2D((2,2))(x)

    x = keras.layers.BatchNormalization()(x)

    

    x = keras.layers.Conv2D(32, (5,5), activation='relu')(x)

    x = keras.layers.MaxPool2D((2,2))(x)

    x = keras.layers.Dropout(.2)(x)

    x = keras.layers.BatchNormalization()(x)

    

    x = keras.layers.Conv2D(64, (5,5), activation='relu')(x)

    x = keras.layers.MaxPool2D((2,2))(x)

    x = keras.layers.BatchNormalization()(x)



    x = keras.layers.Conv2D(128, (3,3), activation='relu')(x)

    x = keras.layers.MaxPool2D((2,2))(x)

    x = keras.layers.Dropout(.4)(x)

    x = keras.layers.BatchNormalization()(x)



    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(500, activation='relu')(x)

    x = keras.layers.Dropout(.5)(x)

    x = keras.layers.Dense(128, activation='relu')(x)

    x = keras.layers.Dropout(.5)(x)

    out = keras.layers.Dense(30)(x)

    

    model = keras.models.Model(inp, out)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model
model = build_model()

model.summary()
images = np.array([image / 255 for image in images])

images = np.expand_dims(images, axis=-1)

print(images.shape)
y = df.values

y /= 96

y = y.astype(np.float32)
hist = model.fit(images, y, 

                 epochs= 2000, batch_size=128,

                 validation_split=.2)#, callbacks=[sched])
import matplotlib.pyplot as plt

%matplotlib inline

epochs = range(1,len(hist.history['loss'])+1)

plt.plot(epochs, hist.history['loss'], 'b', label='Training loss')

plt.plot(epochs, hist.history['val_loss'], 'g', label='Validation loss')

plt.yscale('log')

plt.legend()

plt.show()
test_data = pd.read_csv('../input/test/test.csv')

#preparing test data

timag = []

for i in range(0,1783):

    timg = test_data['Image'][i].split(' ')

    timg = ['0' if x == '' else x for x in timg]

    

    timag.append(timg)
timage_list = np.array(timag, dtype = 'float')

X_test = timage_list.reshape(-1,96,96)

plt.imshow(X_test[0],cmap = 'gray')

plt.scatter(pred[0][::2], pred[0][1::2])

plt.show()
pred = model.predict(np.expand_dims(X_test/255,axis=-1))

pred = (pred * 96) # convert back to non-normalized

print(pred[0])
lookid_data = pd.read_csv('../input/IdLookupTable.csv')

lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId']-1)

pre_list = list(pred)

rowid = lookid_data['RowId']

rowid=list(rowid)
feature = []

for f in list(lookid_data['FeatureName']):

    feature.append(lookid_list.index(f))
preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')

submission = pd.concat([rowid,loc],axis = 1)

submission.to_csv('face_key_detection_submission.csv',

                  index = False)