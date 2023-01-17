%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd



from datetime import datetime

from collections import Counter

import json

import os



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
dim = 256

train_files = []

test_files = []

country_file = ''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        if 'train' in path:

            train_files.append(path)

        elif 'test' in path:

            test_files.append(path)

        elif 'json' in path:

            country_file = path
with open(country_file) as json_file:

    tmp = json.load(json_file)

country = {}

for key in tmp.keys():

  country[int(key)] = tmp[key].split(',')[-1]
train_set = []

for f in train_files:

  idx = int(f.split('/')[6])

  train_set.append([f, country[idx]])

train_set = pd.DataFrame(train_set, columns=['Image','Country'])

    

test_set = []

for f in test_files:

    test_set.append(f)

test_set = pd.DataFrame(test_set, columns=['Image'])
train_data_gen = ImageDataGenerator(rescale = 1./255,

                                    shear_range = 0.2,

                                    zoom_range = 0.2,

                                    horizontal_flip = True)
train_generator = train_data_gen.flow_from_dataframe(

    dataframe = train_set,

    directory="",

    x_col="Image",

    y_col="Country",

    class_mode="categorical",

    target_size=(dim,dim),

    batch_size=32)
num_classes = len(Counter(train_generator.classes).keys())
classifier = Sequential()

classifier.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (dim,dim,3)))

classifier.add(MaxPool2D(pool_size = (2,2)))

classifier.add(Conv2D(64,(3,3),activation = 'relu'))

classifier.add(Conv2D(64,(3,3),activation = 'relu'))

classifier.add(MaxPool2D(pool_size = (2,2)))

classifier.add(Conv2D(128,(3,3),activation = 'relu'))

classifier.add(Conv2D(128,(3,3),activation = 'relu'))

classifier.add(MaxPool2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = num_classes , activation = 'softmax'))

classifier.compile(

    optimizer = 'adam', 

    loss = 'categorical_crossentropy', 

    metrics = ['categorical_accuracy','accuracy']

)

classifier.summary()
classifier.fit_generator(train_generator, epochs = 100, steps_per_epoch = 70)
classes = train_generator.class_indices

print(classes)
inverted_classes = dict(map(reversed, classes.items()))

print(inverted_classes)
from keras.preprocessing import image



Y_pred = []



for idx in range(test_set.shape[0]):

  img = image.load_img(path=test_set['Image'][idx],target_size=(dim,dim,3))

  img = image.img_to_array(img)

  test_img = img.reshape((1,dim,dim,3))

  img_class = classifier.predict_classes(test_img)

  prediction = img_class[0]

  Y_pred.append(prediction)
print(Y_pred)
prediction_classes = [ inverted_classes.get(item,item) for item in Y_pred ]

print(prediction_classes)
predictions = []

for idx in range(test_set.shape[0]):

    predictions.append([test_set['Image'][idx].split('/')[6].split('.')[0],prediction_classes[idx]])

predictions = pd.DataFrame(predictions, columns=['ID','Country'])

predictions['ID'] = predictions['ID'].astype(int)

predictions.sort_values(by=['ID'], inplace=True)

predictions.to_csv(datetime.now().strftime("gallivanters_%Y%m%d_%H%M%S.csv"), index=False)

predictions.head(20)