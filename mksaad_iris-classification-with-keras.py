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
df = pd.read_csv('../input/iris/Iris.csv', index_col='Id')

df
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['Species'] = label_encoder.fit_transform(df['Species'])

df
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers
tf.__version__
df_val = df.sample(frac=0.2, random_state=33)

df_val
df_val.shape
df_train = df.drop(df_val.index)

df_train
# A utility method to create a tf.data dataset from a Pandas Dataframe

def df_to_dataset(dataframe, target, shuffle=True, batch_size=10):

  dataframe = dataframe.copy()

  labels = dataframe.pop(target)

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(dataframe))

  ds = ds.batch(batch_size)

  return ds
train_ds = df_to_dataset(dataframe=df_train, target='Species')

val_ds = df_to_dataset(dataframe=df_val, target='Species')
for b in train_ds.take(1):

    print(b)
from tensorflow import feature_column



feature_columns = []

# numeric cols

for col in df.columns:

  if col == 'Species':

    continue

  feature_columns.append(feature_column.numeric_column(col, dtype=tf.float16)) 

feature_columns
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.DenseFeatures(feature_columns))

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 

             metrics=['accuracy'])
model.evaluate(val_ds)
model.fit(train_ds, epochs=10, validation_data=val_ds)
model.summary()
model.evaluate(val_ds)
model.predict(val_ds)
np.argmax([0.2, 0.9, 0.01])
np.argmax([0.8, 0.19, 0.01])
np.argmax([0.08, 0.19, 0.71])
predictions = np.argmax(model.predict(val_ds), axis=1)

predictions
len(predictions)
df.keys()
df.columns
new_data = {

    'SepalLengthCm': [5.0],

    'SepalWidthCm': [1.2], 

    'PetalLengthCm': [3.5],

    'PetalWidthCm': [0.7]

}

new_data
input_dict = {name: tf.convert_to_tensor([value]) for name, value in new_data.items()}

input_dict
model.predict(input_dict)
np.argmax(model.predict(input_dict))