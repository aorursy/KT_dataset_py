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

print ('done')
train_data_initial = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')

train_data_initial.head()
cols_with_missing = [col for col in train_data_initial.columns

                     if train_data_initial[col].isnull().any()]

for col in cols_with_missing:

    print (col, train_data_initial[col].isnull().sum())
train_data = train_data_initial.copy()
cols_with_big_amount_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']

reduced_train_data = train_data.drop(cols_with_big_amount_missing, axis=1)
s = (reduced_train_data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
object_columns_with_missing_values = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

                                     'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish',

                                     'GarageQual', 'GarageCond']

numeric_columns_with_missing_values = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for col in object_columns_with_missing_values:

    print (col, reduced_train_data[col].value_counts().idxmax())
for col in object_columns_with_missing_values: 

    reduced_train_data[col].fillna(value = reduced_train_data[col].value_counts().idxmax(), inplace = True)

for col in numeric_columns_with_missing_values:

    reduced_train_data[col].fillna(value = reduced_train_data[col].median(), inplace = True)
from sklearn.preprocessing import LabelEncoder

label_reduced_train_data = reduced_train_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_reduced_train_data[col] = label_encoder.fit_transform(reduced_train_data[col])
label_reduced_train_data.info()
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers
tf.__version__
df_val = label_reduced_train_data.sample(frac=0.2, random_state=0)

df_val
df_train = label_reduced_train_data.drop(df_val.index)

df_train
def df_to_dataset(dataframe, target, shuffle=True, batch_size=50):

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

for col in label_reduced_train_data.columns:

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

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae',

                optimizer='Adam',

                metrics=['mae'])
model.fit(train_ds, epochs=400, validation_data=val_ds)
model.summary()
model.evaluate(val_ds)
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

test_data.head()
cols_with_missing_test = [col for col in test_data.columns

                     if test_data[col].isnull().any()]

for col in cols_with_missing_test:

    print (col, test_data[col].isnull().sum())
cols_with_big_amount_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']

reduced_test_data = test_data.drop(cols_with_big_amount_missing, axis=1)
object_columns_test_with_missing_values = ['MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 

                                           'BsmtFinType2', 'GarageType', 'GarageFinish','GarageQual', 'GarageCond',

                                           'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',

                                           'KitchenQual', 'Functional', 'SaleType', 'FireplaceQu','BsmtCond']

numeric_columns_test_with_missing_values = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                                            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 

                                            'GarageCars', 'GarageArea',]
for col in object_columns_test_with_missing_values:

    print (col, reduced_train_data[col].value_counts().idxmax())
for col in object_columns_test_with_missing_values: 

    reduced_test_data[col].fillna(value = train_data_initial[col].value_counts().idxmax(), inplace = True)

for col in numeric_columns_test_with_missing_values:

    reduced_test_data[col].fillna(value = train_data_initial[col].mean(), inplace = True)
s = (reduced_test_data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder

label_reduced_test_data = reduced_test_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_reduced_train_data[col] = label_encoder.fit(train_data_initial[col].astype(str))

    label_reduced_test_data[col] = label_encoder.transform(reduced_test_data[col].astype(str))

    
label_reduced_test_data
preds = []

for i, r in label_reduced_test_data.iterrows():

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in r.items()}

    # print(model.predict(input_dict)[0][0])

    preds.append(model.predict(input_dict)[0][0]) 
preds = pd.Series(preds)

preds
output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': preds})

output
output.to_csv('submission.csv', index=False)

print('done!')