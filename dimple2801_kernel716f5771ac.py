# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division, print_function



# Helper libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



print(tf.__version__)
sf.describe()
sf = pd.read_csv("../input/sf-crime/train.csv", skiprows = 0)

sf.head()
pandas_profiling.ProfileReport(sf)
sf.Category.nunique()
sf.DayOfWeek.value_counts()
sf.Descript.value_counts()
sf.Descript.nunique()
sf.Resolution.nunique()
sf.Resolution.value_counts()
sf['time_of_day']= pd.to_datetime(sf['Dates']).dt.time

sf.head()
sf = sf.assign(time_of_day=pd.cut(pd.to_datetime(sf['Dates']).dt.hour,[-1,5,11,16,20,24],labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late Evening']))
sf.tail()
sf.count()
sf['Category_label'] = LabelEncoder().fit_transform(sf['Category'])

sf.info()
train_data = sf.copy()

train_data.head()
obj_df = sf.select_dtypes(include=['object', 'category'])

obj_df.head()
obj_df = obj_df.drop(obj_df.columns[[0, 1, 2, 5, 6]], axis=1)

obj_df.head()
train_data = pd.get_dummies(obj_df)

train_data.head()
train_data['Category_label'] = sf['Category_label']

train_data.info()
data_array = train_data.to_numpy()

TrainData   = data_array[:, :21]

TrainLabels = data_array[:, [21]]



print(type(TrainData))

TrainLabels = TrainLabels.astype('int')

print(TrainLabels)
print(type(TrainData))

TrainData = TrainData.astype('float')
model = keras.Sequential([

#     keras.layers.Flatten(input_shape=(1, 6)),

#    keras.layers.Dense(128, activation=tf.nn.relu),

    # input layer with 3 input variables not needed because that's implicit

    keras.layers.Dense(11, activation=tf.nn.relu), # first hidden layer

    keras.layers.Dense(39, activation=tf.nn.softmax) # layers for each value of output variable

])
model.compile(

              optimizer='adam', 

#             optimizer='sgd', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(TrainData, TrainLabels, batch_size=5000, epochs=10)
sf_test = pd.read_csv("../input/sf-crime/test.csv", skiprows = 0)

sf_test.head()
sf_test = sf_test.assign(time_of_day=pd.cut(pd.to_datetime(sf_test['Dates']).dt.hour,[-1,5,11,16,20,24],labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late Evening']))

sf_test.head()
test_data = sf_test.drop(sf_test.columns[[0, 1, 4, 5, 6]], axis=1)

test_data.head()

print(test_data.shape)
test_data_encoded = pd.get_dummies(test_data)

test_data_encoded.head()
test_data_array = test_data_encoded.to_numpy()

TestData = test_data_array[:, :21]

predictions = model.predict(TestData)

print(predictions[0])

print(len(predictions))

print(TestData.shape)
predictionsDF = pd.DataFrame(data=predictions)

predictionsDF.insert(loc=0, column='Id', value=sf_test['Id'])

predictionsDF.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(sf['Category'])

le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

print(le_name_mapping)

header = list(le_name_mapping.values())

print(header)
header_1 = ['Id'] + header

print(header_1)
predictionsDF.to_csv("test_submission_final.csv", encoding='utf-8', header=header_1, index=False)