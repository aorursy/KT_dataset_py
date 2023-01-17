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

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:

    zip_ref.extractall('/kaggle/working')

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zip_ref:

    zip_ref.extractall('/kaggle/working')
filenames = os.listdir("/kaggle/working/train")

filenames[0]



categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append('dog')

    else:

        categories.append('cat')



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

        
df.head()

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.20, random_state=13)

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.95):

            print("\nReached 95% accuracy so cancelling training!")

            self.model.stop_training = True



  # Your Code



callbacks = myCallback()
train_datagen = ImageDataGenerator(rescale=1/255)

validation_datagen = ImageDataGenerator(rescale=1/255)



train_generator=train_datagen.flow_from_dataframe(

    train, 

    "/kaggle/working/train/", 

    x_col='filename',

    y_col='category',

    target_size=(128,128),

    class_mode='categorical',

    batch_size=128)



valid_generator=validation_datagen.flow_from_dataframe(

    test, 

    "/kaggle/working/train/", 

    x_col='filename',

    y_col='category',

    target_size=(128,128),

    class_mode='categorical',

    batch_size=128)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128,3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Flatten(),



    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),



    tf.keras.layers.Dense(2, activation='softmax')

])



from tensorflow.keras.optimizers import RMSprop



model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['accuracy'])





history = model.fit(

      train_generator,  

      epochs=15,

      verbose=1,

      validation_data = valid_generator,

      callbacks=[callbacks])
filenames_test = os.listdir("/kaggle/working/test1")

test_df = pd.DataFrame({

    'filename': filenames_test

})
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "/kaggle/working/test1/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(128,128),

    batch_size=128,

    shuffle=False

)
predict=model.predict_generator(test_generator,steps=np.ceil(12500/128))

test_df['category'] = np.argmax(predict, axis=-1)
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)