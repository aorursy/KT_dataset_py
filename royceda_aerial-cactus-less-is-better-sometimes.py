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
! cp -rf /kaggle/input/aerial-cactus-identification/train.csv -d /kaggle/working

! unzip -o /kaggle/input/aerial-cactus-identification/train.zip -d /kaggle/working

! unzip /kaggle/input/aerial-cactus-identification/test.zip -d /kaggle/working
import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import tensorflow as tf

import keras





from keras.datasets import mnist

from sklearn.model_selection import train_test_split



print("tf version : ", tf.__version__)



# GPU test 

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

    raise SystemError('GPU device not found')



print('Found GPU at: {}'.format(device_name))
df = pd.read_csv('train.csv')

#df.has_cactus = np.where(df.has_cactus == 1, 'yes', 'no')

df.sample(3)

df.has_cactus.value_counts().plot.bar()
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical



filename = df.id[10]

print(filename)

image = load_img("./train/"+filename)



plt.imshow(image)
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_datagen = ImageDataGenerator(

    rotation_range=45,

    rescale=1./32,

    zoom_range=0.1,

    horizontal_flip=True,

    vertical_flip=True,

#     width_shift_range=0.1,

#     height_shift_range=0.1



)





#train_datagen = ImageDataGenerator()
BATCH_SIZE = 128

IMAGE_SIZE = (32,32)



INPUT_SHAPE=(32, 32, 3)

BATCH_SIZE=2**10



train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_df, 

    directory="./train",

    x_col='id',

    y_col='has_cactus',

    target_size=IMAGE_SIZE,

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    class_mode="raw"

)





validation_generator = train_datagen.flow_from_dataframe(

    dataframe=validate_df, 

    directory="./train",

    x_col='id',

    y_col='has_cactus',

    target_size=IMAGE_SIZE,

    color_mode='rgb',

    batch_size=BATCH_SIZE,

    class_mode="raw"

)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, AveragePooling2D





model = Sequential([

                    Conv2D(filters=64, kernel_size=(4,4), strides=(1,1), activation='relu', input_shape=(32, 32, 3), padding="same"),

                    BatchNormalization(),

                    AveragePooling2D( pool_size=(3, 3)), 

                    Dropout(0.2),



#                     Conv2D(128, (4, 4), activation='relu',  padding="same"),

#                     BatchNormalization(),

#                     AveragePooling2D( pool_size=(2, 2)), 

#                     #Dropout(0.2),

    

        

#                     Conv2D(64, (2, 2), activation='relu',  padding="same"),

#                     BatchNormalization(),

#                     AveragePooling2D( pool_size=(2, 2)), 

#                     Dropout(0.3),

    

    

#                     Conv2D(32, (2, 2), activation='relu',  padding="same"),

#                     BatchNormalization(),

#                     AveragePooling2D( pool_size=(2, 2)), 

#                     Dropout(0.3),



                    Flatten(),

                    Dense(128, activation='relu'),

                    Dense(64, activation='relu'),

                    Dense(32, activation='relu'),

                    Dropout(0.45),

                    Dense(1, activation='sigmoid')

])





from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=4)

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

callbacks = [earlystop]



model.summary()
%%time

history = model.fit(

    train_generator, 

    epochs=30,

    validation_data=validation_generator,

    #validation_steps=validate_df.shape[0]//BATCH_SIZE,

    #steps_per_epoch=train_df.shape[0]//BATCH_SIZE,

    callbacks=callbacks

)
pd.DataFrame(history.history).plot()
df = pd.DataFrame()

df['id'] = os.listdir('test')

df.head()



from keras.preprocessing import image_dataset_from_directory



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    df,

    "test", 

    x_col='id',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)





pred=model.predict(test_generator)





df['has_cactus'] =np.transpose(pred)[0] #np.argmax(pred, axis=-1)

df.sample(5)
pred
np.transpose(pred)[0]
df.has_cactus.max()
submission = df.copy()

submission.to_csv('submission.csv', index=False)
! ls ../
submission.head()
submission.has_cactus.describe()
! rm -rf train test train.csv