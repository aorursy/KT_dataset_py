# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns



# !pip3 install -q tensorflow==2.0.0-beta1

import tensorflow as tf

print(tf.__version__)



print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
dataset_train = pd.read_csv("../input/train.csv")

dataset_test = pd.read_csv("../input/test.csv")

dataset_train.head()
# show distirbution of sale price

sns.distplot(dataset_train['SalePrice'])
#find all category features

columns_numberic = dataset_train.dtypes[dataset_train.dtypes != 'object'].index

# print(columns_numberic)

columns_string = dataset_train.columns.difference(columns_numberic)

dataset_train[columns_string].head(10)
#encode category feature "neighborhood", 'CentralAir' and show the relative between features

columns_string=['Neighborhood', 'CentralAir']

for column in columns_string:

    label_encoder = preprocessing.LabelEncoder()

    dataset_train[column] = label_encoder.fit_transform(dataset_train[column])

    

corrmat = dataset_train.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=0.8, square=True)
#concat train and test , and remove column 'id' and 'SalePrice'

dataset_all = pd.concat([dataset_train.iloc[:,1:-1], dataset_test.iloc[:,1:]], axis=0, ignore_index=True)



#normalize numberic features

features_numberic = dataset_all.dtypes[dataset_all.dtypes != 'object'].index

dataset_all[features_numberic] = dataset_all[features_numberic].apply(lambda x : ((x - x.mean()) / x.std()))

dataset_all[features_numberic] = dataset_all[features_numberic].fillna(0)



#hot-encode discreted varaibale

dataset_all = pd.get_dummies(dataset_all, dummy_na=True)



#show samples

dataset_all.head()
#split train and label dataset

train_labels_origin = dataset_train['SalePrice']



train_labels_mean = train_labels_origin.mean()

train_labels_std = train_labels_origin.std()



#normalize train label

train_labels = (train_labels_origin - train_labels_mean ) / train_labels_std



n_train = train_labels.shape[0]

print("train_labels.shape=", n_train)

train_X = dataset_all.iloc[0:n_train, :]

print("train_X.shape=", train_X.shape)



test_X = dataset_all.iloc[n_train: , :]

print("test_X.shape=", test_X.shape)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64, input_shape=(train_X.shape[1], ), activation="relu"))

model.add(tf.keras.layers.Dropout(0.72))

model.add(tf.keras.layers.Dense(1))



optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer, loss="mse")

# model.compile(optimizer, loss=tf.keras.losses.MeanSquaredLogarithmicError()) #rmsle

model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2)

history = model.fit(train_X, train_labels, batch_size=128, epochs=100, validation_split=0.4, callbacks=[early_stopping])
df_his = pd.DataFrame(history.history)

df_his.plot()

plt.show()
#output normalized value

results = model.predict(test_X) 

#restore real value

results  = results * train_labels_std + train_labels_mean 
#show predicated values of "Id, SalePrice"

results = results.reshape(1,-1)[0]

dataset_test["SalePrice"] = pd.Series(results)

submission = pd.concat([dataset_test['Id'], dataset_test['SalePrice']], axis = 1)

submission.head()