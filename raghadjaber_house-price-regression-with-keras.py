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
df=pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col= 'Id')

df
miss_obj=[]

miss_flt=[]



for col in df.columns:

    if df[col].isnull().sum()>0:

        if df[col].dtypes == 'object':

            miss_obj.append(df[col].name)

        elif df[col].dtypes == 'float64':

            miss_flt.append(df[col].name)     

        print(col,': missing =', df[col].isnull().sum())



print('miss_obj = ',miss_obj)

print('miss_flt = ', miss_flt)
for cols in miss_flt:

    df[cols].fillna(df[cols].mean(), inplace=True)
for cols in miss_obj:

    df[cols].fillna(str(df[cols].mode()), inplace=True)
for col in df.columns:

    if df[col].isnull().sum()>0:

        print(col,': missing =', df[col].isnull().sum())

obj_cols=[]



for col in df.columns:

    if df[col].dtypes == 'object':

        obj_cols.append(df[col].name)



print('obj_cols = ',obj_cols)
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_df = df.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in obj_cols:

    label_df[col] = label_encoder.fit_transform(df[col])

label_df
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
tf.__version__
label_df_val= label_df.sample(frac=0.2, random_state= 9)

label_df_val
label_df_val.shape
label_df_train= label_df.drop(label_df_val.index)

label_df_train
def df_to_dataset(dataframe, target, shuffle=True, batch_size=100):

    dataframe = dataframe.copy()

    labels = dataframe.pop(target)

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    if shuffle:

        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(batch_size)

    return ds
train_ds = df_to_dataset(dataframe=label_df_train, target='SalePrice')

val_ds = df_to_dataset(dataframe=label_df_val, target='SalePrice')
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

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mae', 

             metrics=['mae','mse'])
model.evaluate(val_ds)
model.fit(train_ds, epochs=10, validation_data=val_ds)
model.summary()
model.evaluate(val_ds)
model.predict(val_ds)
X_test = pd.read_csv( '../input/home-data-for-ml-course/test.csv', index_col= 'Id')
miss_obj_X=[]

miss_flt_X=[]



for col in X_test.columns:

    if X_test[col].isnull().sum()>0:

        if X_test[col].dtypes == 'object':

            miss_obj_X.append(X_test[col].name)

        elif X_test[col].dtypes == 'float64':

            miss_flt_X.append(X_test[col].name)     

        print(col,': missing =', X_test[col].isnull().sum())



print('miss_obj_X = ',miss_obj_X)

print('miss_flt_X = ', miss_flt_X)
for cols in miss_flt_X:

    X_test[cols].fillna(label_df[cols].mean(), inplace=True)
for cols in miss_obj_X:

    X_test[cols].fillna(str(label_df[cols].mode()), inplace=True)
for col in X_test.columns:

    if X_test[col].isnull().sum()>0:

        print(col,': missing =', X_test[col].isnull().sum())
obj_cols_X=[]



for col in X_test.columns:

    if X_test[col].dtypes == 'object':

        obj_cols_X.append(X_test[col].name)



print('obj_cols_X = ',obj_cols_X)
label_X_test = X_test.copy()



# Apply label encoder to each column with categorical data

label_encoder=LabelEncoder()

label_encoder.fit(label_df.drop('SalePrice', axis=1))

for col in obj_cols_X:

    label_X_test[col] = label_encoder.transform(X_test[col])

label_X_test