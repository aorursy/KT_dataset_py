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
import cv2

new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))
i = 0
for f, l in data_entry_subset[:9].values:
    img = cv2.imread('../input/nihdata/images_001/images/{}'.format(f))
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    i += 1
    
plt.show()
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  
    zoom_range=0.2,        
    horizontal_flip=True,
    validation_split=0.2) 

def get_flow_from_dataframe(generator, 
                            dataframe, 
                            subset,
                            image_shape=(150, 150),
                            batch_size=32):
    
    train_generator_1 = generator.flow_from_dataframe(dataframe, target_size=image_shape,
                                                      x_col='Image Index',
                                                      y_col='Finding Labels',
                                                      class_mode='categorical',
                                                      directory = '../input/nihdata/images_001/images',
                                                      batch_size=batch_size,
                                                      classes = labels,
                                                      subset=subset)

    train_generator_2 = generator.flow_from_dataframe(dataframe, target_size=image_shape,
                                                      x_col='Image Index',
                                                      y_col='Finding Labels',
                                                      class_mode='categorical',
                                                      directory = '../input/nihdata/images_002/images',
                                                      batch_size=batch_size,
                                                      classes = labels,
                                                      subset=subset)
    
    train_generator_3 = generator.flow_from_dataframe(dataframe, target_size=image_shape,
                                                      x_col='Image Index',
                                                      y_col='Finding Labels',
                                                      class_mode='categorical',
                                                      directory = '../input/nihdata/images_003/images',
                                                      batch_size=batch_size,
                                                      classes = labels,
                                                      subset=subset)
    
    train_generator_4 = generator.flow_from_dataframe(dataframe, target_size=image_shape,
                                                      x_col='Image Index',
                                                      y_col='Finding Labels',
                                                      class_mode='categorical',
                                                      directory = '../input/nihdata/images_004/images',
                                                      batch_size=batch_size,
                                                      classes = labels,
                                                      subset=subset)
    
    while True:
        x_1 = train_generator_1.next()
        x_2 = train_generator_2.next()
        x_3 = train_generator_3.next()
        x_4 = train_generator_4.next()

        yield np.concatenate((x_1[0], x_2[0], x_3[0], x_4[0]), axis = 0), np.concatenate((x_1[1], x_2[1], x_3[1], x_4[1]), axis = 0)
train_gen = get_flow_from_dataframe(generator=datagen, 
                                    dataframe=data_entry_subset, 
                                    subset = 'training',
                                    image_shape=(150, 150),
                                    batch_size=32)

val_gen = get_flow_from_dataframe(generator=datagen, 
                                    dataframe=data_entry_subset, 
                                    subset = 'validation',
                                    image_shape=(150, 150),
                                    batch_size=32)
generator = datagen.flow_from_dataframe(data_entry_subset, target_size=(150,150),
                                                      x_col='Image Index',
                                                      y_col='Finding Labels',
                                                      class_mode='categorical',
                                                      directory = '../input/nihdata/images_001/images',
                                                      batch_size=32,
                                                      classes = labels)

generator.class_indices
class_weights_index = {
 1: 50.985951008645536,
 5: 56.25476947535771,
 4: 10.628294660959675,
 10: 2.3448418680936367,
 7: 623.5110132158591,
 8: 7.114557152910425,
 9: 24.478900034590108,
 11: 22.356183857210553,
 0: 12.244744355048015,
 14: 26.695020746887966,
 12: 41.81299852289513,
 13: 98.9077568134172,
 6: 83.94839857651246,
 3: 61.45766391663048,
 2: 30.327190914934647
}
from keras import layers
from keras import models
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.regularizers import l2

#Trying just 1 layer

model = models.Sequential()
model.add(layers.Conv2D(64, (4,4), padding = 'same',
                        activation='relu', input_shape=(150, 150, 3)))
#model.add(layers.MaxPooling2D((2, 2)))

#model.add(layers.Conv2D(128, (4,4), activation='relu'))##

#model.add(layers.Flatten()) 
#model.add(layers.Dense(1024, activation='relu')) ##
model.add(layers.Dense(15, activation='sigmoid'))
from keras import optimizers

model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
step_per_train = 28000//32
step_per_val = 6999//32

history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = step_per_train, 
    validation_data  = val_gen,
    validation_steps = step_per_val,
    class_weight = class_weights_index,
    use_multiprocessing = True,
    epochs = 2)#was 5
#Trying 2 layers

model = models.Sequential()
model.add(layers.Conv2D(64, (4,4), padding = 'same',
                        activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (4,4), activation='relu'))##

model.add(layers.Flatten()) 
#model.add(layers.Dense(1024, activation='relu')) ##
model.add(layers.Dense(15, activation='sigmoid'))
step_per_train = 28000//32
step_per_val = 6999//32

history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = step_per_train, 
    validation_data  = val_gen,
    validation_steps = step_per_val,
    class_weight = class_weights_index,
    use_multiprocessing = True,
    epochs = 2)#was 5
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



model = models.Sequential()
model.add(layers.Conv2D(64, (4,4), padding = 'same',
                        activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (4,4), activation='relu'))##
model.add(layers.MaxPooling2D((2, 2)))##

model.add(layers.Conv2D(128, (4, 4), activation='relu',kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))##
tf.keras.layers.Dropout(.5, input_shape=(150, 150, 3))##

model.add(layers.Conv2D(128, (4,4), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (4,4), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (4,4), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten()) 
model.add(layers.Dense(1024, activation='relu')) ##
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(15, activation='sigmoid'))
'''from keras import layers
from keras import models
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.regularizers import l2

from keras import backend as K
K.tensorflow_backend.set_image_dim_ordering('th')




model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding = 'same',
                        activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
#model.add(Conv2D(32, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
tf.keras.layers.Dropout(.5, input_shape=(150, 150, 3))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))#added
model.add(layers.MaxPooling2D((2, 2)))#added


model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(15, activation='sigmoid'))''';
from keras import optimizers

model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
# Model Training
#This session is to fit the data into the model.
#add more dropouts
#improving weight decay
#adding more neurons?
step_per_train = 28000//32
step_per_val = 6999//32

history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = step_per_train, 
    validation_data  = val_gen,
    validation_steps = step_per_val,
    class_weight = class_weights_index,
    use_multiprocessing = True,
    epochs = 2)#was 5
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
acc

