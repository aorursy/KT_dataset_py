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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')
train.columns

test.columns
#new_train=train.drop(columns='SalePrice')
train.columns

for i in train.columns:    

    print(i ,': ',train[i].isnull().sum())
cols_with_missing_train = [col for col in train.columns

                     if train[col].isnull().any()]

cols_with_missing_test = [col for col in test.columns

                     if test[col].isnull().any()]

all_missing_columns = cols_with_missing_train + cols_with_missing_test

print(len(all_missing_columns))

train.drop(all_missing_columns, axis=1,inplace=True)

test.drop(all_missing_columns, axis=1,inplace=True)
filteredColumns =train.dtypes[train.dtypes == np.object]

listOfColumnNames = list(filteredColumns.index)

print(listOfColumnNames)

train.drop(listOfColumnNames, axis=1,inplace=True)

test.drop(listOfColumnNames, axis=1,inplace=True)
for i in train.columns:    

    print(i ,': ',train[i].isnull().sum())
train.SalePrice
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers
tf.__version__

df_val = train.sample(frac=0.2, random_state=33)

df_val
df_val.shape
df_train = train.drop(df_val.index)

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
df_train
train_ds = df_to_dataset(dataframe=df_train, target='SalePrice')

val_ds = df_to_dataset(dataframe=df_val, target='SalePrice')
for b in train_ds.take(1):

    print(b)
from tensorflow import feature_column

feature_columns = []

for col in train.columns:

    if col == 'SalePrice':

        continue

    feature_columns.append(feature_column.numeric_column(col, dtype=tf.float16)) 

feature_columns
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.DenseFeatures(feature_columns))

model.add(tf.keras.layers.Dense(32, activation='relu'))

#model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='mae', 

             metrics=['mae'])
model.evaluate(val_ds)

model.fit(train_ds, epochs=100, validation_data=val_ds)
model.summary()
model.evaluate(val_ds)
preds = []

for i, r in test.iterrows():

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in r.items()}

    # print(model.predict(input_dict)[0][0])

    preds.append(model.predict(input_dict)[0][0])   
preds = pd.Series(preds)

preds
# Save test predictions to file

output = pd.DataFrame({'Id': test.index,

                       'SalePrice': preds})

output
output.to_csv('submission.csv', index=False)

print('done!')