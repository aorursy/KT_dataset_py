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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator( rescale = 1.0/255. )



train_dir = "/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/train/"

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=32,

                                                    class_mode='categorical',

                                                    target_size=(256, 256))
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(4, activation='softmax')

])



model.summary()
model_mod = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),

    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

    tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(4, activation='softmax')

])



model_mod.summary()
from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from tensorflow.keras.optimizers import RMSprop



model_mod.compile(loss='categorical_crossentropy',

                  optimizer=RMSprop(lr=0.001),

                  metrics=[f1])
history = model_mod.fit(train_generator,

                        steps_per_epoch=28,

                        epochs=10,

                        verbose=2)
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing import image



result = []

test_dir = '/kaggle/input/testpraktikum1/test/test/'

for filename in os.listdir(test_dir) :

    file_dir = test_dir + filename

    img = load_img(file_dir, target_size=(256, 256))

    img_arr = image.img_to_array(img)

    img_arr = np.expand_dims(img_arr, axis=0)

    cls = model_mod.predict(img_arr)

    result.append({'id': filename, 'label': np.argmax(cls)})
df = pd.DataFrame(result)

df.head()
df.to_csv('P1_13517001_13517016_13517055.csv', index=False)



os.chdir(r'../working')

from IPython.display import FileLink

FileLink(r'P1_13517001_13517016_13517055.csv')