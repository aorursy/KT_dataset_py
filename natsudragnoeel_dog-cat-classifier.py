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
import os 

import zipfile

local_zip = '/kaggle/input/dogs-vs-cats/train.zip'

zip_ref=zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/kaggle/tmp')

zip_ref.close()
base_dir = '/kaggle/tmp'

train_dir = os.path.join(base_dir,'train')
train_dir_frames = os.listdir(train_dir)

print(train_dir_frames[:5])
filenames = os.listdir("/kaggle/tmp/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.tail(5)
print("tortal training Images = ",len(os.listdir('/kaggle/tmp/train')))
%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



nrow = 4

ncols = 4

pic_index = 0
df['category'].value_counts().plot.bar()
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

from sklearn.model_selection import train_test_split
import random

sample = random.choice(filenames)

print(sample)

img = mpimg.imread('/kaggle/tmp/train/'+sample)

plt.imshow(img)

model = tf.keras.models.Sequential([

    

    tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape = (150,150,3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    

    

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.20),

    

    

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    

    

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.20),

    

    

    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Dropout(0.25),

    

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.50),

    tf.keras.layers.Dense(1,activation='softmax')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.0001),

             loss='categorical_crossentropy',metrics=['accuracy'])
print(df.head(10))

df['category'] = df['category'].replace({0:'cat',1:'dog'})

print(df.head(10))
train_data , test_data = train_test_split(df, test_size = 0.2,random_state = 30)



train_data = train_data.reset_index(drop= True)

test_data = test_data.reset_index(drop=True)

print(train_data.head(10))

print(test_data.head(10))



train_data['category'].value_counts().plot.bar()
train_len = train_data.shape[0]

test_len = test_data.shape[0]

batch_size = 20
train_datagen = ImageDataGenerator(rescale = 1./255,

                              rotation_range = 30,

                              shear_range = .2,

                              zoom_range = .2,

                              horizontal_flip = True,

                              width_shift_range = .15,

                              height_shift_range = .15)

train_generator = train_datagen.flow_from_dataframe(train_data,'/kaggle/tmp/train/',

                                                   x_col = 'filename',

                                                   y_col = 'category',

                                                   target_size = (150, 150),

                                                   class_mode = 'categorical',

                                                   batch_size = batch_size)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_dataframe(test_data, '/kaggle/tmp/train/',

                                                 x_col = 'filename',

                                                 y_col = 'category',

                                                 target_size = (150, 150),

                                                 class_mode = 'categorical',

                                                 batch_size = batch_size)
example_data = train_data.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(example_data,'/kaggle/tmp/train',

                                                     x_col = 'filename',

                                                     y_col = 'category',

                                                     target_size = (150, 150),

                                                     class_mode = 'categorical')
plt.figure(figsize=(16,16))

for i in range(0,16):

    plt.subplot(4,4 , i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
fastrun = False 

epochs = 3 if fastrun else 20

history = model.fit_generator(train_generator,

                             epochs = epochs,

                             validation_data = test_generator,

                             validation_steps = test_len//batch_size,

                             steps_per_epoch = train_len//batch_size)
model.save_weights('model.h5')
fig, (ax1,ax2)=plt.subplots(2,1,figsize=(12,12))

ax1.plot(history.history['loss'], color='b',label = 'Training_loss')

ax1.plot(history.history['val_loss'], color='r',label='validation_loss')

ax1.set_xticks(np.arange(1,epochs,1))

ax1.set_yticks(np.arange(0,1,0.1))





ax2.plot(history.history['accuracy'],color='b', label = 'Training_accuracy')

ax2.plot(history.history['val_accuracy'], color='r',label='Validation_accuracy')

ax2.set_xticks(np.arange(1,epochs,1))

legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
import os 

import zipfile

local_zip = '/kaggle/input/dogs-vs-cats/test1.zip'

zip_ref=zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/kaggle/tmp')

zip_ref.close()



base_dir = '/kaggle/tmp'

test_dir = os.path.join(base_dir,'test1')



test_dir_frames = os.listdir(test_dir)

print(test_dir_frames[:5])



test_filenames = os.listdir("/kaggle/tmp/test1")

test_df = pd.DataFrame({'filename': test_filenames})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale = 1./255)

test_generator = test_gen.flow_from_dataframe(test_df, '/kaggle/tmp/test1/',

                                            x_col = 'filename',

                                            y_col = None,

                                            class_mode = None,

                                            target_size = (150,150),

                                            batch_size = batch_size,

                                            shuffle = False)
predict = model.predict_generator(test_generator,steps = np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis = -1)

label_map = dict((v,k) for k, v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] =  test_df['category'].replace({'dog':1, 'cat': 0})

test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize = (12,24))



for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img('/kaggle/tmp/test1/'+filename, target_size = (150,150))

    plt.subplot(6,3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + ' ( ' + " {} ".format(category)+" ) " )

plt.tight_layout()

plt.show()