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
traindf = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

testdf = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
traindf
testdf
num_col = [col for col in traindf.columns if traindf[col].dtype != 'object']

cat_col = [col for col in traindf.columns if traindf[col].dtype == 'object']

(len(num_col) + len(cat_col)) == traindf.shape[1]

len(num_col), len(cat_col), traindf.shape[1]
bad_cols = []

for col in cat_col:

    if set(traindf[col].unique()) != set(testdf[col].unique()):

        bad_cols.append(col)

print(bad_cols, len(bad_cols))
traindf.drop(columns=bad_cols, axis=1, inplace=True)

testdf.drop(columns=bad_cols, axis=1, inplace=True)
# update cat 

cat_col = [col for col in traindf.columns if traindf[col].dtype == 'object']

len(cat_col)
import tensorflow as tf 

from tensorflow import keras

from tensorflow import feature_column

tf.__version__
train_num = traindf[num_col] 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

train_num_imputed = pd.DataFrame(imputer.fit_transform(train_num))

train_num_imputed.columns = train_num.columns 

train_num_imputed
df_val = train_num_imputed.sample(frac=0.2, random_state=1033)

df_train = train_num_imputed.drop(df_val.index)
df_train
# A utility method to create a tf.data dataset from a Pandas Dataframe

def df_to_dataset(dataframe, target, shuffle=True, batch_size=10):

  my_df = dataframe.copy()

  labels = my_df.pop(target)

  ds = tf.data.Dataset.from_tensor_slices((dict(my_df), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(dataframe))

  ds = ds.batch(batch_size)

  return ds
train_ds = df_to_dataset(dataframe=df_train, target='SalePrice', batch_size=100)

val_ds = df_to_dataset(dataframe=df_val,  target='SalePrice', batch_size=100)
for b in train_ds.take(1):

    print(b)
from tensorflow import feature_column



features = []

for col in train_num_imputed.columns:

    if col == 'SalePrice':

        continue 

    features.append(feature_column.numeric_column(col))

features
model = keras.models.Sequential()

model.add(keras.layers.DenseFeatures(features))

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss=keras.losses.mean_absolute_error, 

             metrics=['mae', 'mse'])
model.fit(train_ds, validation_data=val_ds, epochs=100, verbose=0)
model.summary()
model.evaluate(val_ds)
df_train.shape
test_num = testdf.select_dtypes(exclude=['object'])

test_num.shape
test_imputer = SimpleImputer()

test_imputer.fit(df_train.drop('SalePrice', axis=1))

test_num_imputed = pd.DataFrame(test_imputer.transform(test_num))

test_num_imputed.columns = test_num.columns 

test_num_imputed
test_df_y = pd.concat([test_num_imputed, pd.DataFrame({'SalePrice': pd.Series([0] * 1459 )})], axis=1)

test_df_y
test_ds = df_to_dataset(dataframe=test_df_y,  target='SalePrice', batch_size=100)
preds = model.predict(test_ds)

preds.shape
preds_list = []

for r in preds:

    preds_list.append((r[0]))

preds_list = pd.Series(preds_list)

preds_list
# Save test predictions to file

output = pd.DataFrame({'Id': testdf.index,

                       'SalePrice': preds_list})

output
output.to_csv('submission.csv', index=False)

print('done!')