# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.io import imread

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pathlib import Path

# Define path to the data directory

data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')



# Path to train directory (Fancy pathlib...no more os.path!!)

train_dir = data_dir / 'train'



# Path to validation directory

val_dir = data_dir / 'val'



# Path to test directory

test_dir = data_dir / 'test'
# Get the path to the normal and pneumonia sub-directories

normal_cases_dir = train_dir / 'NORMAL'

pneumonia_cases_dir = train_dir / 'PNEUMONIA'



# Get the list of all the images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

train_data = []



# Go through all the normal cases. The label for these cases will be 0

for img in normal_cases:

    train_data.append((img,0))



# Go through all the pneumonia cases. The label for these cases will be 1

for img in pneumonia_cases:

    train_data.append((img, 1))



# Get a pandas dataframe from the data we have in our list 

train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)



# Shuffle the data 

train_data = train_data.sample(frac=1.).reset_index(drop=True)



# How the dataframe looks like?

train_data.head()
# Get few samples for both the classes

pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()

normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()



# Concat the data in a single list and del the above two list

samples = pneumonia_samples + normal_samples

del pneumonia_samples, normal_samples



# Plot the data 

f, ax = plt.subplots(2,5, figsize=(30,10))

for i in range(10):

    img = imread(samples[i])

    # ax[i//5, i%5].imshow(img, cmap='gray')

    ax[i//5, i%5].imshow(img)

    if i<5:

        ax[i//5, i%5].set_title("Pneumonia", color="white")

    else:

        ax[i//5, i%5].set_title("Normal", color="white")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
# Get the path to the normal and pneumonia sub-directories

val_normal_cases_dir = val_dir / 'NORMAL'

val_pneumonia_cases_dir = val_dir / 'PNEUMONIA'



# Get the list of all the images

val_normal_cases = val_normal_cases_dir.glob('*.jpeg')

val_pneumonia_cases = val_pneumonia_cases_dir.glob('*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

val_data = []



# Go through all the normal cases. The label for these cases will be 0

for img in val_normal_cases:

    val_data.append((img,0))



# Go through all the pneumonia cases. The label for these cases will be 1

for img in val_pneumonia_cases:

    val_data.append((img, 1))

    

# Get a pandas dataframe from the data we have in our list 

val_data = pd.DataFrame(val_data, columns=['image', 'label'],index=None)



# Shuffle the data 

val_data = val_data.sample(frac=1.).reset_index(drop=True)



# How the dataframe looks like?

print (len(val_data))
# Only run once

# Since validation data is only 16, I will push it 10% by throwing some training data into the validation data

from sklearn.utils import shuffle

train_data = shuffle(train_data)



def oversample(diff, data):

    ids = np.arange(len(data))

    choices = np.random.choice(ids, diff)

    return data.iloc[choices]



val_length = 0.1 * len(train_data)

diff = round(val_length - len(val_data))

oversample_data = oversample(diff, train_data)

train_length = len(train_data)



train_data.drop(oversample_data.index, inplace=True)

val_data = val_data.append(oversample_data)

print (len(val_data), len(train_data))
print ("Normal:{}\nPneumonia:{}".format(train_data.label.value_counts()[0], train_data.label.value_counts()[1]))
# Make balance sampling by Random OverSampling

normal_data = train_data[train_data['label'] == 0]

ids = np.arange(len(normal_data))

diff = len(train_data[train_data['label'] == 1]) - len(normal_data)

choices = np.random.choice(ids, diff)



oversample_data = normal_data.iloc[choices]

train_data = train_data.append(oversample_data)



print ("Normal:{}\nPneumonia:{}".format(train_data.label.value_counts()[0], train_data.label.value_counts()[1]))
import tensorflow as tf

from tensorflow.keras import applications

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import initializers





# Default from Keras page

image_width, image_height= 224, 224

model = applications.ResNet50V2(weights= "imagenet", include_top=False, pooling='max', input_shape=(image_height, image_width,3))



#creating a fully connected layer. 

x=model.output



# try kernel initializator

x=Dense(1024, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(512, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(256, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)



predictions = Dense(1, activation="sigmoid")(x) #try sigmoid



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

mc = ModelCheckpoint('base_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



model_final =Model(inputs=model.input, outputs=predictions)

model_final.compile(optimizer ='SGD',loss='binary_crossentropy', metrics =['accuracy'])

#model_final.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
# model_final.summary()
# Change type to fit into ImageDataGenerator



for col in train_data.columns:

    train_data[col] = train_data[col].astype(str) 



for col in val_data.columns:

    val_data[col] = val_data[col].astype(str)
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True,

                                   fill_mode="nearest",

                                   width_shift_range=0.3,

                                   height_shift_range=0.3,

                                   rotation_range=30)



val_datagen = ImageDataGenerator(rescale = 1./255,

                                    horizontal_flip = True,

                                    fill_mode = "nearest",

                                    zoom_range = 0.3,

                                    width_shift_range = 0.3,

                                    height_shift_range=0.3,

                                    rotation_range=30)



train_generator=train_datagen.flow_from_dataframe(dataframe=train_data, x_col="image", y_col="label", class_mode="binary", target_size=(224,224), batch_size=32)

val_generator=val_datagen.flow_from_dataframe(dataframe=val_data, x_col='image', y_col='label', class_mode='binary', target_size=(224,224), batch_size=32)
hist = model_final.fit_generator(train_generator,epochs =15, validation_data = val_generator,callbacks=[es, mc])
from matplotlib import pyplot

# plot training history

pyplot.plot(hist.history['loss'], label='train')

pyplot.plot(hist.history['val_loss'], label='val')

pyplot.legend()

pyplot.show()
# Get the path to the normal and pneumonia sub-directories

test_normal_cases_dir = test_dir / 'NORMAL'

test_pneumonia_cases_dir = test_dir / 'PNEUMONIA'



# Get the list of all the images

test_normal_cases = test_normal_cases_dir.glob('*.jpeg')

test_pneumonia_cases = test_pneumonia_cases_dir.glob('*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

test_data = []



# Go through all the normal cases. The label for these cases will be 0

for img in test_normal_cases:

    test_data.append((img,0))



# Go through all the pneumonia cases. The label for these cases will be 1

for img in test_pneumonia_cases:

    test_data.append((img, 1))

    

# Get a pandas dataframe from the data we have in our list 

test_data = pd.DataFrame(test_data, columns=['image', 'label'],index=None)



# Shuffle the data 

test_data = test_data.sample(frac=1.).reset_index(drop=True)



# How the dataframe looks like?

print (len(test_data))
from tensorflow.keras.models import load_model



test_data['image'] = test_data['image'].astype(str)

test_datagen = ImageDataGenerator(rescale = 1./255)



test_generator=test_datagen.flow_from_dataframe(

    dataframe=test_data,

    x_col='image',

    batch_size=32,

    class_mode=None,

    shuffle=False,

    seed=42,

    pickle_safe = True,

    use_multiprocessing=False,

    workers = 0,

    target_size=(224,224))



test_generator.reset()

model_final = load_model('base_model.h5')

pred = model_final.predict_generator(test_generator,verbose=1)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred = np.round(pred)

test_data['label'] = test_data['label'].astype(float)

print('Accuracy Score')

print(accuracy_score(test_data['label'], pred))



print('Confusion Matrix')

print(confusion_matrix(test_data['label'], pred))



print('Classification Report')

target_names = ['Normal', 'Pneumonia']

print(classification_report(test_data['label'], pred, target_names=target_names))
# for col in test_data.columns:

#     test_data[col] = test_data[col].astype(str)

    

# test_datagen = ImageDataGenerator(rescale = 1./255)

# model_final.metrics_name = ['normal', 'pneumonia']

# test_generator=test_datagen.flow_from_dataframe(

#     dataframe=test_data,

#     x_col='image',

#     y_col='label',

#     batch_size=32,

#     class_mode='binary',

#     seed=42,

#     use_multiprocessing=False,

#     shuffle=False,

#     pickle_safe = True,

#     workers = 0,

#     target_size=(224,224))



# score = model_final.evaluate_generator(test_generator)

# print("loss: %.3f - acc: %.3f" % (score[0], score[1]))
## Using a heavier DenseNet architecture

model_heavy = applications.DenseNet201(weights= "imagenet", include_top=False, pooling='max', input_shape=(image_height, image_width,3))



#creating a fully connected layer. 

x=model_heavy.output



# try kernel initializator

x=Dense(1024, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(512, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(256, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)



predictions = Dense(1, activation="sigmoid")(x) #try sigmoid



model_heavy =Model(inputs=model_heavy.input, outputs=predictions)

model_heavy.compile(optimizer ='SGD',loss='binary_crossentropy', metrics =['accuracy'])

#model_heavy.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
# model_heavy.summary()
# Additional of a dense layer

model = applications.ResNet50V2(weights= "imagenet", include_top=False, pooling='max', input_shape=(image_height, image_width,3))



#creating a fully connected layer. 

x=model.output



# try kernel initializator

x=Dense(1024, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(512, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(256, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

# Add additional dense layer

x=Dense(128, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)



predictions = Dense(1, activation="sigmoid")(x) #try sigmoid



model_dense =Model(inputs=model.input, outputs=predictions)

model_dense.compile(optimizer ='SGD',loss='binary_crossentropy', metrics =['accuracy'])

#model_dense.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
# model_dense.summary()
# Change pooling

model = applications.ResNet50V2(weights= "imagenet", include_top=False, pooling='avg', input_shape=(image_height, image_width,3))



#creating a fully connected layer. 

x=model.output

# print (x.shape)

# x=GlobalAveragePooling2D(x)

# try kernel initializator

x=Dense(1024, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(512, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)

x=Dense(256, activation="relu",kernel_initializer=initializers.glorot_uniform(seed=0))(x)

x=Dropout(0.5)(x)



predictions = Dense(1, activation="sigmoid")(x) #try sigmoid



model_pool =Model(inputs=model.input, outputs=predictions)

model_pool.compile(optimizer ='SGD',loss='binary_crossentropy', metrics =['accuracy'])

#model_pool.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
# model_pool.summary()
for modelz in [model_heavy, model_dense, model_pool]:

    if modelz == model_heavy:

        name = 'model_heavy'

    elif modelz == model_dense:

        name = 'model_dense'

    else:

        name = 'model_pool'

    

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    mc = ModelCheckpoint('{}.h5'.format(name), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



    hist = modelz.fit_generator(train_generator,epochs = 15, validation_data = val_generator,callbacks=[es, mc])

    

    # plot training history

    pyplot.plot(hist.history['loss'], label='train')

    pyplot.plot(hist.history['val_loss'], label='val')

    pyplot.legend()

    pyplot.show()

    

    test_generator.reset()

    loadmodel = load_model('{}.h5'.format(name))

    pred = loadmodel.predict_generator(test_generator)

    

    pred = np.round(pred)

    test_data['label'] = test_data['label'].astype(float)

    print('Accuracy Score for {}'.format(name))

    print(accuracy_score(test_data['label'], pred))



    print('Confusion Matrix for {}'.format(name))

    print(confusion_matrix(test_data['label'], pred))



    print('Classification Report'.format(name))

    target_names = ['Normal', 'Pneumonia']

    print(classification_report(test_data['label'], pred, target_names=target_names))