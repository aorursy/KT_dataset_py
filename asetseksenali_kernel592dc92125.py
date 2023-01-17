# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Dense, Flatten

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

from numpy import asarray

import scipy

import gc

tqdm.pandas()
id_lookup = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

train = pd.read_csv('/kaggle/input/facial-keypoints-detection/training/training.csv')

clear_data = train.dropna()
def get_images(dataframe):

    return np.stack(dataframe.progress_apply(lambda row: np.array(row.Image.split()).reshape(96, 96).astype('int') / 255., axis=1)) / 96.
x_train = get_images(clear_data)

y_train = clear_data.iloc[:, :-1].values.flatten().reshape((-1, 30)) / 96.

x_train = np.expand_dims(x_train, axis=3)
model = Sequential()

model.add(Conv2D(32, kernel_size=(4, 4), activation = 'elu', input_shape=(96, 96, 1), padding='valid'))

model.add(MaxPooling2D())

model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', padding='valid'))

model.add(MaxPooling2D())

model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(2, 2), activation='elu', padding='valid'))

model.add(MaxPooling2D())

model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(1, 1), activation='elu', padding='valid'))

model.add(MaxPooling2D())

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1000, activation='elu'))

model.add(Dropout(0.5))

model.add(Dense(1000, activation='elu'))

model.add(Dropout(0.6))

model.add(Dense(30, activation='linear')) # We add dense layers for processing input
model.compile(loss='mse',

             optimizer='adam',

             metrics=['mae'])
history = model.fit(x=x_train,y=y_train, batch_size=64, epochs=150, verbose=1, validation_split=0.2)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
test = pd.read_csv('/kaggle/input/facial-keypoints-detection/test/test.csv')

test_images = np.expand_dims(get_images(test), axis=3)

result = model.predict(test_images) * 96.
res = pd.DataFrame(result, columns=[train.columns[:-1]])

res.index += 1

res = res.unstack()

kek = res.swaplevel().sort_index().reset_index()

kek = kek.rename({'level_0':"ImageId", 'level_1': "FeatureName"}, axis=1)

r = id_lookup.merge(kek, how = 'left')
result = pd.DataFrame({

    "RowId": r.RowId,

    "Location": r.iloc[:, -1]

})
plt.imshow(test_images[874].reshape((96, 96)))

plt.plot(res.unstack().iloc[:, 874].values[::2], res.unstack().iloc[:, 874].values[1::2], 'r.')
result.loc[result.Location > 96]
result.to_csv('result1.csv', index=False)