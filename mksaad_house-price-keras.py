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
X = pd.read_csv('../input/home-data-for-ml-course/train.csv', 

                 index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

X
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

len(cat_cols)
X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', 

                     index_col='Id')

X_test 
bad_cols = [ col for col in cat_cols if set(X[col].unique()) != set(X_test[col].unique()) ]

bad_cols                                       
len(bad_cols)
X.drop(columns=bad_cols, axis=1, inplace=True)

X_test.drop(columns=bad_cols, axis=1, inplace=True)
X
#update cat_cols

cat_cols = [col for col in X.columns if X[col].dtype == 'object']

len(cat_cols)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='most_frequent')

X_imputed = pd.DataFrame(imputer.fit_transform(X))

X_imputed.columns = X.columns



test_imputer = SimpleImputer(strategy='most_frequent')

test_imputer.fit(X.drop('SalePrice', axis=1))

X_test_imputed = pd.DataFrame(test_imputer.transform(X_test))

X_test_imputed.columns = X_test.columns
X_imputed
X_imputed.isna().sum().any()
X_test_imputed
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X_imputed_label = X_imputed.copy()

X_test_imputed_label = X_test_imputed.copy()

for col in cat_cols:

    X_imputed_label[col] = label_encoder.fit_transform(X_imputed[col])

    X_test_imputed_label[col] = label_encoder.transform(X_test_imputed[col])
X_imputed_label
X_test_imputed_label
X_imputed_label.dtypes
X_imputed_label = X_imputed_label.apply(pd.to_numeric) 

X_test_imputed_label = X_test_imputed_label.apply(pd.to_numeric) 
X_imputed_label.dtypes
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers

tf.__version__
X_imputed_label
df_val = X_imputed_label.sample(frac=0.2, random_state=33)

df_val
df_train = X_imputed_label.drop(df_val.index)

df_train
# A utility method to create a tf.data dataset from a Pandas Dataframe

def df_to_dataset(dataframe, target, shuffle=True, batch_size=10):

  my_df = dataframe.copy()

  labels = my_df.pop(target)

  ds = tf.data.Dataset.from_tensor_slices((dict(my_df), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(my_df))

  ds = ds.batch(batch_size)

  return ds
df_train.isna().sum().any()
train_ds = df_to_dataset(dataframe=df_train, batch_size=100, target='SalePrice')

val_ds = df_to_dataset(dataframe=df_val, batch_size=100, target='SalePrice')
from tensorflow import feature_column



feature_columns = []

# numeric cols

for col in X_imputed_label.columns:

  if col == 'SalePrice':

    continue

  feature_columns.append(feature_column.numeric_column(col, dtype=tf.float16)) 

feature_columns
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.DenseFeatures(feature_columns))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.compile(optimizer='adam', loss='mae', 

              metrics=['mae', 'mse'])
model.fit(train_ds, epochs=400, validation_data=val_ds, verbose=0)
model.summary()
model.evaluate(val_ds)
X_test_imputed_label
input_dict = {name: tf.convert_to_tensor([value]) for name, value in X_test_imputed_label.loc[0].items()}

input_dict
model.predict(input_dict)
preds = []

for i, r in X_test_imputed_label.iterrows():

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in r.items()}

    # print(model.predict(input_dict)[0][0])

    preds.append(model.predict(input_dict)[0][0])   
preds = pd.Series(preds)

preds
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output
output.to_csv('submission.csv', index=False)

print('done!')