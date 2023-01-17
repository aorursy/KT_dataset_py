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

from tensorflow import keras 

tf.__version__
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

train_data
y_train = train_data['label']

x_train = train_data.drop('label', axis=1)
y_train.value_counts()
x_train = x_train / 255.0 
x_train
x_train.shape
x_train_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in x_train.iterrows()] ] )
x_train_np.shape
model = tf.keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(32, activation='relu'))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))
model.build()

model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(x_train_np, y_train, epochs=50, batch_size=100, validation_split=0.1)
test_data = pd.read_csv('../input/digit-recognizer/test.csv')

test_data
test_data = test_data / 255.0

test_data
x_test_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in test_data.iterrows()] ] )
x_test_np.shape
model.predict(x_test_np)
preds = pd.Series(np.argmax(model.predict(x_test_np), axis=1).tolist())
out_df = pd.DataFrame({

    'ImageId': pd.Series(range(1,28001)),

    'Label': preds

})

out_df
out_df.to_csv('my_submission.csv', index=False)