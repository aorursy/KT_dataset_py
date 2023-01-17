# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
SEED = 257



image_width = 100

image_height = 100



TRAIN_DIR = '../input/train/train/'

TEST_DIR = '../input/test/test/'



categories = ['hot dog', 'not hot dog']
import os



import numpy as np

import pandas as pd



from matplotlib.image import imread

import matplotlib.pyplot as plt

%matplotlib inline
X, y = [], []



for category in categories:

    category_dir = os.path.join(TRAIN_DIR, category)

    for image_path in os.listdir(category_dir):

        X.append(imread(os.path.join(category_dir, image_path)))

        y.append(category)
from tensorflow.keras.utils import to_categorical



labels = to_categorical([1 if x == 'hot dog' else 0 for x in y])

data = np.array(X).reshape(len(X), image_width, image_height, 3)
from sklearn.model_selection import train_test_split



data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.25, random_state=SEED)
# Почему-то на kaggle.com нет оптимизации под GPU.

# На colab.research.google.com конфигурить сэссию не нужно.

import tensorflow as tf

config = tf.ConfigProto()

config.intra_op_parallelism_threads = 128

config.inter_op_parallelism_threads = 128

tf.Session(config=config)



model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(2, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=0.0, amsgrad=False)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data_train, labels_train, 

          batch_size=128,

          epochs=16,

          validation_data=(data_test, labels_test))
from sklearn.metrics import roc_auc_score



roc_auc_score(labels_test, model.predict_proba(data_test))
leaderboard_X = []

leaderboard_filenames = []



for image_path in os.listdir(TEST_DIR):

    leaderboard_X.append(imread(os.path.join(TEST_DIR, image_path)))

    leaderboard_filenames.append(image_path)
data_control = np.array(leaderboard_X).reshape(len(leaderboard_X), image_width, image_height, 3)

leadeboard_predictions = model.predict(data_control, verbose=1)[:,1]
submission = pd.DataFrame(

    {

        'image_id': leaderboard_filenames, 

        'image_hot_dog_probability': leadeboard_predictions

    }

)

submission.head()
submission.to_csv('submit.csv', index=False)