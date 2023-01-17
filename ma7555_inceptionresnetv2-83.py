# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras

from matplotlib.image import imread

from matplotlib import pyplot as plt

from progressbar import ProgressBar

from collections import Counter

from sklearn import preprocessing

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

imagenames = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    if dirname.endswith(r'train/train'):

        print(dirname)

        train_dir = dirname

    for filename in filenames:

        if filename.endswith('.jpg') and dirname.endswith(r'train/train'):

            imagenames.append(filename)

print(len(imagenames))

# Any results you write to the current directory are saved as output.
# Checking images sizes..

pbar = ProgressBar()

img_sizes = []

for imagename in pbar(imagenames):

        img_sizes.append(imread(os.path.join(train_dir, imagename)).shape)

Counter(img_sizes).most_common(6)
doggo_images = np.empty((len(imagenames), 224, 224, 3), dtype=np.float16) # Large Array, A MemoryError could be raised

doggo_targets = np.empty((len(imagenames)), dtype=np.uint8)



labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')

le = preprocessing.LabelEncoder()

labels['target'] = le.fit_transform(labels.breed)
pbar = ProgressBar()

for i in pbar(range(len(imagenames))): 

    doggo_images[i] = np.expand_dims(

                        keras.preprocessing.image.img_to_array(

                            keras.preprocessing.image.load_img(

                                os.path.join(train_dir, imagenames[i]),

                                    target_size=(224, 224))), axis=0) / 255.0

    doggo_targets[i] = labels[labels.id == imagenames[i].rstrip('.jpg')].target.values[0]

print(doggo_images.shape)
class_dogs = []

breed = []

for i in np.sort(np.unique(doggo_targets)):

    class_dogs.append(doggo_images[doggo_targets == i][np.random.randint(50)])

    breed.append(i)
# Some dog breeds look pretty similar

plt.figure(figsize=(11,11))

for i in range(len(np.unique(doggo_targets))):

    plt.subplot(11, 11, i+1) # plot index can not be 0

    plt.imshow(class_dogs[i].astype(np.float32))

    plt.axis('off')

plt.show()
keras.backend.clear_session()



model_base = keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))



model = keras.models.Sequential()

for layer in model_base.layers:

    layer.trainable = False



model.add(model_base)

model.add(keras.layers.GlobalAveragePooling2D())

model.add(keras.layers.Dense(120, activation='softmax'))

    

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(doggo_images, doggo_targets, validation_split=0.05, epochs=20, batch_size=1024)
pbar = ProgressBar()

testnames = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    if dirname.endswith(r'test/test'):

        test_dir = dirname

    for filename in filenames:

        if filename.endswith('.jpg') and dirname.endswith(r'test/test'):

            testnames.append(filename)



doggo_test_images = np.empty((len(testnames), 224, 224, 3), dtype=np.float16)



for i in pbar(range(len(testnames))): 

    doggo_test_images[i] = np.expand_dims(

                        keras.preprocessing.image.img_to_array(

                            keras.preprocessing.image.load_img(

                                os.path.join(test_dir, testnames[i]),

                                    target_size=(224, 224))), axis=0) / 255.0

predictions = pd.DataFrame(model.predict(doggo_test_images))

predictions.columns = list(le.inverse_transform([int(col) for col in predictions.columns]))

predictions['id'] = [testname.rstrip('.jpg') for testname in testnames]
predictions.head()
predictions.to_csv('submission.csv', index=False)