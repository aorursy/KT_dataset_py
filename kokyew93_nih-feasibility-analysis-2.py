import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # for pattern matching, like grep in r

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
data_entry = pd.read_csv('../input/nihdata/Data_Entry_2017.csv')

data_entry.head()
data_entry_subset = data_entry.loc[:, 'Image Index':'Finding Labels']

data_entry_subset.head()
split = pd.DataFrame(data_entry_subset['Finding Labels'].str.split('|').tolist())



temp = []

for i in split:

    temp.append(split[i].unique())



flatten = pd.DataFrame(temp).values.flatten()



unique = []

for x in flatten:

    if x not in unique:

        unique.append(x)



labels = list(filter(None, unique))

labels
data_entry_subset["Finding Labels"] = data_entry_subset["Finding Labels"].apply(lambda x:x.split("|"))

data_entry_subset.head()
from collections import Counter

labels_count = Counter(label for lbs in data_entry_subset["Finding Labels"] for label in lbs)



labels_count
total_count = sum(labels_count.values())

class_weights = {cls: total_count / count for cls, count in labels_count.items()}



class_weights

labels_index = pd.DataFrame(labels)

labels_index['index'] = labels_index.index

labels_index
class_weights_index = {

 0: 50.985951008645536,

 7: 56.25476947535771,

 5: 10.628294660959675,

 1: 2.3448418680936367,

 2: 623.5110132158591,

 4: 7.114557152910425,

 3: 24.478900034590108,

 6: 22.356183857210553,

 8: 12.244744355048015,

 10: 26.695020746887966,

 9: 41.81299852289513,

 14: 98.9077568134172,

 11: 83.94839857651246,

 13: 61.45766391663048,

 12: 30.327190914934647

}
import cv2



new_style = {'grid': False}

plt.rc('axes', **new_style)

_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))

i = 0

for f, l in data_entry_subset[:9].values:

    img = cv2.imread('../input/nihdata/images_001/images/{}'.format(f))

    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))

    #ax[i // 4, i % 4].show()

    i += 1

    

plt.show()
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,  

    zoom_range=0.2,        

    horizontal_flip=True,

    validation_split=0.2)  



train_generator = datagen.flow_from_dataframe(

    dataframe = data_entry_subset,

    x_col = 'Image Index',

    y_col = 'Finding Labels',

    directory = '../input/nihdata/images_001/images',

    target_size = (150,150),

    batch_size = 32,

    class_mode = 'categorical',

    classes = labels,

    subset = 'training')



val_generator = datagen.flow_from_dataframe(

    dataframe = data_entry_subset,

    x_col = 'Image Index',

    y_col = 'Finding Labels',

    directory = '../input/nihdata/images_001/images',

    target_size = (150,150),

    batch_size = 32,

    class_mode = 'categorical',

    classes = labels,

    subset = 'validation')
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), padding = 'same',

                        activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(15, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
history = model.fit_generator(

    train_generator, 

    steps_per_epoch  = 10, 

    validation_data  = val_generator,

    validation_steps = 50,

    class_weight = class_weights_index,

    epochs = 10)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

fig = plt.figure(figsize=(16,9))



plt.subplot(1, 2, 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()
model.save('multi_label.h5')
from keras.models import load_model

from keras.preprocessing import image



model = load_model('multi_label.h5')



val_generator.reset()

pred = model.predict_generator(val_generator, verbose = 1)
predictions = []

pred_threshold = (pred > 0.25)

class_indices = train_generator.class_indices

class_indices = dict((v, k) for k, v in class_indices.items())



for i in pred_threshold:

    outcome = []

    for index, cls in enumerate(i):

        if cls:

            outcome.append(class_indices[index])

    predictions.append(",".join(outcome))



patient_id = val_generator.filenames

results = pd.DataFrame({"Filename": patient_id,

                       "Classifications": predictions})



results.head()
import cv2



new_style = {'grid': False}

plt.rc('axes', **new_style)



for f, l in data_entry_subset.iloc[[4]].values:

    img = cv2.imread('../input/nihdata/images_001/images/{}'.format(f))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title(('{} - {}'.format(f, l)))

    

plt.show()