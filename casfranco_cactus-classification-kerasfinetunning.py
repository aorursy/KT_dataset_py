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
# More imports

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
# extracting files

!cp /kaggle/input/aerial-cactus-identification/*.* /kaggle/working/

!unzip '*.zip'
# Visualize the dataset structure

train_df = pd.read_csv('train.csv')

train_df.head()
# Number of elements per class

print(len(train_df[train_df.has_cactus == 1]))

print(len(train_df[train_df.has_cactus == 0]))
# numerical visual inspection

train_df.has_cactus.value_counts().plot.barh()
# Pie visual inspection

train_df.has_cactus.value_counts().plot.pie(autopct='%.2f')
# New Balanced dataset

df_noncactus = train_df[train_df["has_cactus"] == 0]

df_cactus = train_df[train_df["has_cactus"] == 1]



np.random.seed(seed=21)

random_index = np.random.permutation(len(df_noncactus))

train_samples = np.floor(len(random_index)*0.8).astype(int)

training_idx, validation_idx = random_index[:train_samples], random_index[train_samples:]



train_df_cactus, validation_df_cactus = df_cactus.iloc[training_idx], df_cactus.iloc[validation_idx]  

train_df_noncactus, validation_df_noncactus = df_noncactus.iloc[training_idx], df_noncactus.iloc[validation_idx]  



train_df_set = pd.concat([train_df_cactus, train_df_noncactus], ignore_index=True, sort=False)

validation_df_set = pd.concat([validation_df_cactus, validation_df_noncactus], ignore_index=True, sort=False)





#train_df_set = train_df_set.reindex(np.random.permutation(train_df_set.index))

train_df_set = train_df_set.iloc[np.random.permutation(np.arange(len(train_df_set)))]

validation_df_set = validation_df_set.iloc[np.random.permutation(np.arange(len(validation_df_set)))]



train_df_set.reset_index(drop=True, inplace=True)

validation_df_set.reset_index(drop=True, inplace=True)

#train_df_set.head(n=10)

train_df_set.has_cactus.value_counts().plot.pie(autopct='%.2f')
# labels should be strings to work with keras generator

train_df_set.has_cactus=train_df_set.has_cactus.astype(str)

validation_df_set.has_cactus=validation_df_set.has_cactus.astype(str)
# Keras generators - flow_from_dataframe



BATCH_SIZE = 32

IMAGE_SIZE = (32,32)



train_gen=ImageDataGenerator(

    rescale=1./255, 

    rotation_range=20,  

    zoom_range = 0.1, 

    width_shift_range=0.2,  

    height_shift_range=0.2, 

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest')  



val_gen=ImageDataGenerator(

    rescale=1./255)



train_generator=train_gen.flow_from_dataframe(

    x_col='id',                                  

    y_col='has_cactus',

    dataframe=train_df_set, 

    directory="train", 

    class_mode='binary',

    #classes=None,

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    target_size=IMAGE_SIZE,

    shuffle=True,

    seed=2019)



val_generator = val_gen.flow_from_dataframe(

    x_col='id',                                  

    y_col='has_cactus',

    dataframe=validation_df_set, 

    directory="train", 

    class_mode='binary',

    #classes=None,

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    target_size=IMAGE_SIZE,

    shuffle=True,

    seed=2019)
from keras.applications.vgg16 import VGG16

from keras import models

from keras import layers

from keras import optimizers



conv_base = VGG16(weights='imagenet',

                  include_top=False,

                  input_shape=(32, 32, 3))
# Adding final layers

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid')) # for binary classification



model.summary()
# Freeze the conv layers for transfer learning.

conv_base.trainable = False



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=2e-5),

              metrics=['acc'])
# Callbacks

from keras.callbacks.callbacks import EarlyStopping,ModelCheckpoint



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

mc = ModelCheckpoint('best_vgg16.h5', monitor='val_loss', mode='min', verbose=1)
history = model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    epochs=100,

    validation_data=val_generator,

    validation_steps=len(val_generator),

    callbacks=[es,mc],

    verbose=0)
def plot_learning_curves(history):

    import matplotlib.pyplot as plt



    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(acc))



    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()
plot_learning_curves(history)
#load best model



model = load_model('best_vgg16.h5')
conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['acc'])



history = model.fit_generator(

    train_generator,

    steps_per_epoch=len(train_generator),

    epochs=100,

    validation_data=val_generator,

    validation_steps=len(val_generator),

    callbacks=[es,mc],

    verbose=1)
model = load_model('best_vgg16.h5')
test_submission = pd.read_csv("sample_submission.csv")

test_submission.head()
BATCH_SIZE = 32



test_gen=ImageDataGenerator(

    rescale=1./255)



test_generator = test_gen.flow_from_dataframe(

    x_col='id',                                  

    #y_col='has_cactus',

    dataframe=test_submission, 

    directory="test", 

    class_mode=None,

    #classes=None,

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    target_size=IMAGE_SIZE,

    shuffle=False,

    seed=2019)
probabilities = model.predict_generator(test_generator)

final_pred = (probabilities > 0.5).astype(np.int)
test_submission['has_cactus'] = final_pred

test_submission.to_csv('sample_submission.csv', index = False)

test_submission.to_csv('submission.csv', index = False)