

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from zipfile import ZipFile

with ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zipObj:

    zipObj.extractall('/kaggle/working/train/')

with ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zipObj:

    zipObj.extractall('/kaggle/working/test1')
for dirname, _, filenames in os.walk('/kaggle/working/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

img = plt.imread('/kaggle/working/train/train/cat.4067.jpg')
img.shape
plt.imshow(img)
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img
train_data_dir = '/kaggle/working/train/train/'

test_data_dir = '/kaggle/working/test1/test1/'
img = plt.imread(train_data_dir + '/dog.101.jpg')

plt.imshow(img)
filenames = os.listdir("/kaggle/working/train/train/")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)
filenames[:10]
df= pd.DataFrame({

    'filename':filenames,

    'category': categories

})

df.head()
df['category'].value_counts().plot.bar()
import random

sample = random.choice(filenames)

image = load_img("/kaggle/working/train/train/"+sample)

plt.imshow(image)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_width, img_height = 128, 128

batch_size = 32

image_size = (img_width, img_height)

n_channels = 3
print(img_width, img_height)
df['category'] = df['category'].replace({0: 'cat', 1:'dog'})
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=10)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df.category.value_counts().plot.bar()
validate_df.category.value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size = 32
train_image_generator = ImageDataGenerator(rescale=1.0/255)

validation_datagen = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_dataframe(train_df,

                                                           "kaggle/working/train/train/",

                                                           x_col = 'filename',

                                                           y_col = 'category',

                                                           target_size = image_size,

                                                           batch_size = batch_size,

                                                           class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_dataframe(validate_df,

                                                              'kaggle/working/train/train/',

                                                              x_col='filename',

                                                              y_col='category',

                                                              target_size = image_size,

                                                              class_mode='categorical',

                                                              batch_size=batch_size)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, n_channels)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(64, (3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(128, (3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))





model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(rate=0.40))

model.add(Dense(2, activation='softmax'))



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Early Stop

earlystop = EarlyStopping(patience=10)
# Learning rate Reduction

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',

                                           patience=2,

                                           verbose=1,

                                           factor=0.5,

                                           min_lr=0.0001)
callbacks = [earlystop, learning_rate_reduction]
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "/kaggle/working/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=image_size,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "/kaggle/working/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=image_size,

    class_mode='categorical',

    batch_size=batch_size

)
epochs = 10



history = model.fit_generator(train_generator,

                             epochs=epochs,

                             validation_data = validation_generator,

                             validation_steps=total_validate//batch_size,

                             steps_per_epoch=total_train//batch_size,

                             callbacks=callbacks)
model.save_weights("model.h5")
history.history
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,12))

ax1.plot(history.history['loss'], color='b', label='Training Loss')

ax1.plot(history.history['val_loss'], color='r', label='Validation Loss')

ax1.set_xticks(np.arange(0, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy')

ax2.plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')

ax2.set_xticks(np.arange(0, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_filenames = os.listdir("/kaggle/working/test1/test1/")

test_df = pd.DataFrame(

    {

        'filename': test_filenames

    }

)

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)



test_generator = test_gen.flow_from_dataframe(test_df,

                                             "/kaggle/working/test1/test1/",

                                             x_col='filename',

                                             y_col=None,

                                             class_mode=None,

                                             target_size= image_size,

                                             batch_size=batch_size,

                                             shuffle=False)
predictions = model.predict_generator(test_generator)
len(predictions), np.ceil(nb_samples//batch_size)
predictions
len(test_df)
test_df['category'] = np.argmax(predictions, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({'dog':1, 'cat':0})
sample_test = test_df.tail(18)



plt.figure(figsize=(12, 24))

index = 0

for i, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("/kaggle/working/test1/test1/"+filename, target_size=image_size)

    plt.subplot(6, 3, index+1)

    index += 1

    plt.imshow(img)

    plt.xlabel(filename + '('+ f'{category}' +')')

plt.tight_layout()

plt.show()    
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)