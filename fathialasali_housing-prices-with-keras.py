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
df = pd.read_csv('../input/home-data-for-ml-course/train.csv' , index_col='Id')

df
cols_with_missing_train = [col for col in df.columns

                     if df[col].isnull().any()]

print(cols_with_missing_train)

print('----------------------')

print(set(cols_with_missing_train))

df.drop(cols_with_missing_train, axis=1,inplace=True)
df.info
df.info
filteredColumns = df.dtypes[df.dtypes == np.object]



print(filteredColumns.index)

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

df.drop(listOfColumnNames, axis=1,inplace=True)
df.dtypes
df.shape
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers
tf.__version__
df_val =df.sample(frac=0.2, random_state=23)

df_val
df_val.shape
df_train =df.drop(df_val.index)

df_train
def df_to_dataset(dataframe, target, shuffle=True, batch_size= 50):

  dataframe = dataframe.copy()

  labels = dataframe.pop(target)

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(dataframe))

  ds = ds.batch(batch_size)

  return ds
train_ds =df_to_dataset(dataframe=df_train, target='SalePrice')

val_ds =df_to_dataset(dataframe=df_val, target='SalePrice')
for b in train_ds.take(1):

    print(b)
from tensorflow import feature_column



feature_columns = []

# numeric cols

for col in df.columns:

  if col == 'SalePrice':

    continue

  feature_columns.append(feature_column.numeric_column(col, dtype=tf.float16)) 

feature_columns
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.DenseFeatures(feature_columns))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',

            optimizer=optimizer,

            metrics=['mae', 'mse'])

#return model
model.fit(train_ds, epochs=50, validation_data=val_ds)
model.summary()
model.evaluate(val_ds)
model.predict(val_ds)