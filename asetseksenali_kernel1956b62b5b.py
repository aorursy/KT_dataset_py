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
filenames = []

for i in range(7):

    path = '/kaggle/input/cat-dataset/cats/CAT_0' + str(i)

    for file in os.listdir(path):

        if not file in filenames and file.endswith('.jpg'):

            filenames.append(file)
annotations, images = [], []



for f in tqdm(filenames):

    counter = 0

    break_second = False

    for i in range(7):

        path = '/kaggle/input/cat-dataset/cats/CAT_0' + str(i)

        for file in os.listdir(path):

            if file.endswith(f + '.cat'):

                annotations.append(path + '/' + file)

                counter += 1

            elif file.endswith(f):

                images.append(path + '/' + file)

                counter += 1

            if counter == 2:

                break_second = True

                break

        if break_second:

            break
data = pd.DataFrame(

    data = {

        'image': images,

        'annotation': annotations

    }, index = filenames)

points = data.progress_apply(lambda row: open(row['annotation']).read(), axis=1)



del filenames

del images

del annotations

gc.collect()
points[0]
data['points'] = points.map(lambda row: [int(row.split()[i]) for i in range(1, 19)])

del points

gc.collect()
def get_percentage(image_url, points):

    image = load_img(image_url)

    width = image.width

    height = image.height

    result = []

    for i in range(len(points)):

        if i % 2 == 0:

            result.append(points[i] / width)

        else:

            result.append(points[i] / height)

    return result
data['points'] = data.progress_apply(lambda row: get_percentage(row['image'], row['points']), axis=1)

gc.collect()
image_arrays = np.empty((data.shape[0], 299, 299, 3), dtype='float16')

target_size=(299, 299)

counter = 0

for _, row in tqdm(data.iterrows(), total=data.shape[0]):

    image_arrays[counter] = asarray(Image.open(row['image']).resize(target_size))

    counter += 1
image_arrays /= 255.

gc.collect()
plt.imshow(image_arrays[1].astype('float32'))
import sys

def sizeof_fmt(num, suffix='B'):

    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f %s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f %s%s" % (num, 'Yi', suffix)



for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),

                         key= lambda x: -x[1])[:10]:

    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
getsizeof(image_arrays) / 1024 / 1024 / 1024
y_train = np.vstack(data['points'].values)
del data

gc.collect()
model = Sequential()

model.add(InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))) # As a basis of model we use InceptionResNetV2

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))

model.add(Dense(18, activation='linear')) # We add dense layers for processing input

model.layers[0].trainable=False # Don't process gradient for ready model
model.compile(loss='mse',

             optimizer='adam',

             metrics=['mae'])
model.fit(x=image_arrays,y=y_train, batch_size=64, epochs=3, verbose=1, validation_split=0.2)
model.save('my_model.h5')