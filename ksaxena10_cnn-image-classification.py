# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Read data with labels

data_labels = pd.read_csv("/kaggle/input/hasyv2-dataset-friend-of-mnist/HASYv2/hasy-data-labels.csv")
# Display top 5 records

data_labels.head()
# Dsiplay all unique labels supported for symbols

data_labels["latex"].unique()
# Extracting data only for selected labels

df = data_labels.loc[data_labels["latex"].isin(['0','1','2','3','4','5','6','7','8','9','x','+','-','/']) ]
# Shape of numpy array after slicing of numpy array based on labels

df.shape
# Images per labels

df.latex.value_counts().plot.bar()
# Sample of dataframe after selecting data for requirted labels

df.head()
# Preparing training data as numpy array of vectors from raw images

import numpy as np

from PIL import Image

from skimage import io, color



base = "/kaggle/input/hasyv2-dataset-friend-of-mnist/HASYv2/"

listOfB = []

for items in df['path'].iteritems(): 

    pil_im = Image.open(base +items[1], 'r')

    array1 = np.asarray(pil_im)/255.

    listOfB.append(array1)

B = np.array(listOfB) 

B.reshape(1826, 32, 32, 3)

B.shape
train_x = B
# Indexing raw labels to indexes of lists

df['latex'].nunique()

classes = ['0','1','2','3','4','5','6','7','8','9','x','+','-','/']

len(classes)
# Preparing categorical labels for multiclass classification (categorical crossentropy)

train_y = df['latex']

listOfk = []

for items in df['latex'].iteritems():

    intermediate = np.zeros(14)

    index = classes.index(items[1])

    intermediate[index] = 1

    listOfk.append(intermediate)

categorical_labels = np.array(listOfk)    

print(categorical_labels.shape)    
train_y = categorical_labels

import keras

from keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping
# preparing neural network model using keras API

model = keras.models.Sequential()

model.add(Conv2D(32, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(32,32,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(14, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
# Data generator add randomness to existing data.

#data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=.1)

## consider using this for more variety

data_generator_with_aug = keras.preprocessing.image.ImageDataGenerator(validation_split=.1, width_shift_range=.1,

                                                                       height_shift_range=.1, rotation_range=20,

                                                                       zoom_range=.1, shear_range=.1)



# if already ran this above, no need to do it again

# X, y = img_label_load(train_data_path)

# print("X.shape: ", X.shape)



training_data_generator = data_generator_with_aug.flow(train_x, train_y, subset='training')

validation_data_generator = data_generator_with_aug.flow(train_x, train_y, subset='validation')

history = model.fit_generator(training_data_generator, 

                              steps_per_epoch=1300, epochs=5,   #5 for demo, use 25 for better results

                              validation_data=validation_data_generator, 

                              validation_steps= len(train_x) / 500,

                              callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.1, patience=3)])
model.save('./maths_model.h5')
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server