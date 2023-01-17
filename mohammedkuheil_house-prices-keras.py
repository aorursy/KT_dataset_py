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
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

df
df.columns

cols_with_missing = [[col,df[col].isna().sum()] for col in df.columns

                     if df[col].isnull().any()]

cols_with_missing
drop_col = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

new_df = df.drop(drop_col, axis=1)

new_df
imputed_col = ['LotFrontage','MasVnrArea','GarageYrBlt']

impu_df = df[imputed_col]

impu_df
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_df = pd.DataFrame(my_imputer.fit_transform(impu_df))

# Imputation removed column names; put them back

imputed_df.columns = impu_df.columns

imputed_df

new_df.update(imputed_df)

new_df[imputed_col].isnull().sum()
# Get list of categorical variables

s = (new_df.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)

from sklearn.preprocessing import LabelEncoder

cols = object_cols

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(new_df[c].values)) 

    new_df[c] = lbl.transform(list(new_df[c].values))
y = new_df['SalePrice']

X = new_df.drop(columns=['SalePrice'])

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2)
df = new_df

df
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers

tf.__version__
df_val = df.sample(frac=0.2, random_state=33)

df_val
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
train_ds = df_to_dataset(dataframe=df_train, target='SalePrice')

val_ds = df_to_dataset(dataframe=df_val, target='SalePrice')
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

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mae',metrics=['mae'])
model.evaluate(val_ds)
model.fit(train_ds, epochs=400, validation_data=val_ds)
model.summary()
model.evaluate(val_ds)
predictions = list (model.predict(val_ds))

predictions
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

df_test
df_test = df_test.drop(drop_col, axis=1)

df_test
object_columns_test = df_test.select_dtypes(include='object')

object_columns_test
# Get list of categorical variables

s1 = (df_test.dtypes == 'object')

object_columns_test = list(s1[s1].index)



print("Categorical variables:")

print(object_columns_test)
from sklearn.preprocessing import LabelEncoder

cols = object_columns_test

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(df_test[c].values)) 

    df_test[c] = lbl.transform(list(df_test[c].values))

imputer = SimpleImputer(strategy='most_frequent')

df_test_final = pd.DataFrame(imputer.fit_transform(df_test))

df_test_final.columns = df_test.columns
cols_with_missing = [col for col in df_test.columns if df_test_final[col].isnull().any()]

cols_with_missing
input_dict = {name: tf.convert_to_tensor([value]) for name, value in df_test.loc[1461].items()}

input_dict
preds = []

for i, r in df_test_final.iterrows():

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in r.items()}

    # print(model.predict(input_dict)[0][0])

    preds.append(model.predict(input_dict)[0][0])
preds = pd.Series(preds)

preds
# Save test predictions to file

output = pd.DataFrame({'Id': df_test.index,

                       'SalePrice': preds})

output