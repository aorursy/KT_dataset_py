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
data_train = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')

data_test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')

data_train.tail()
for i in data_train:

    if data_train[i].isnull().sum() > 0:

        print(i,' - ',data_train[i].isnull().sum())
cols_with_missing = [col for col in data_train.columns if data_train[col].isnull().any()]



# Fill in the lines below: drop columns in training and validation data

reduced_X_train = data_train.drop(cols_with_missing, axis=1,inplace=True)

reduced_X_valid = data_test.drop(cols_with_missing, axis=1,inplace=True)
data_train
set(data_train) - set(data_test)

SalePrice = data_train['SalePrice']

data_train.drop(columns='SalePrice',axis=1,inplace=True)

data_train
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer(strategy='most_frequent')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(data_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(data_test))



# Imputation removed column names; put them back

imputed_X_train.columns = data_train.columns

imputed_X_valid.columns = data_test.columns
data_train = imputed_X_train

data_test = imputed_X_valid
s = (data_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = data_train.copy()

label_X_valid = data_test.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_encoder.fit(pd.concat([data_train[col], data_test[col]], axis=0, sort=False))

    label_X_train[col] = label_encoder.transform(data_train[col])

    label_X_valid[col] = label_encoder.transform(data_test[col])
data_train = label_X_train

data_test = label_X_valid

data_test
data_train.insert(0, 'SalePrice',  list(SalePrice))
data_train.SalePrice
import tensorflow as tf

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers
df_val = data_train.sample(frac=0.2, random_state=33)

df_val
df_val.shape
df_train = data_train.drop(df_val.index)

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

for col in data_train.columns:

    if col == 'SalePrice':

        #print(col)

        continue

    feature_columns.append(feature_column.numeric_column(col, dtype=tf.float16)) 

feature_columns
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.DenseFeatures(feature_columns))

model.add(tf.keras.layers.Dense(120, activation='relu'))

model.add(tf.keras.layers.Dense(120, activation='relu'))

model.add(tf.keras.layers.Dense(120, activation='relu'))

model.add(tf.keras.layers.Dense(120, activation='relu'))

model.add(tf.keras.layers.Dense(1))



#model.add(tf.keras.layers.Dense(3, activation='softmax'))
#model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.evaluate(val_ds)
model.fit(train_ds, epochs=100, validation_data=val_ds)
model.evaluate(val_ds)
input_dict = {name: tf.convert_to_tensor([value]) for name, value in data_test.loc[0].items()}

input_dict
model.predict(input_dict)
preds = []

for i, r in data_test.iterrows():

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in r.items()}

    # print(model.predict(input_dict)[0][0])

    preds.append(model.predict(input_dict)[0][0])  

    print(i,'> ',r)
preds = pd.Series(preds)

preds
samplesubmission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

output = pd.DataFrame({'Id': samplesubmission.Id, 'SalePrice': preds})

output.to_csv('submission.csv', index=False)

print('Done')

##########################

#output = pd.DataFrame({'Id': data_test.index,

 #                      'SalePrice': preds})

#output
#output.to_csv('submission.csv', index=False)

#print('done!')