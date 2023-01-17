# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
raw_train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
raw_train


from sklearn.preprocessing import StandardScaler 

cols = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']

df_train = raw_train[cols]

# Create dummy values

df_train = pd.get_dummies(df_train)

#filling NA's with the mean of the column:

df_train = df_train.fillna(df_train.mean())

# Always standard scale the data before using NN

scale = StandardScaler()

X_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]

X_train = scale.fit_transform(X_train)

# Y is just the 'SalePrice' column

y_train = df_train['SalePrice'].values
import tensorflow as tf


model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.Dense(1)

])







model.compile(optimizer='adam',

              loss='mse',

              metrics=['mse', 'mae'])



model.summary()



history=model.fit(X_train,y_train, epochs=10)

history

#df_test=test[cols]

X_test = test[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]

testdata = X_test.values

test_predictions = model.predict(testdata).flatten()

submission = pd.read_csv('../input/sample_submission.csv')

submission['SalePrice'] = test_predictions

submission.isnull().any().any()

submission.to_csv("my_submission.csv",index=False)
submission