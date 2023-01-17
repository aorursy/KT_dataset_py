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
from sklearn.preprocessing import MinMaxScaler



import tensorflow as tf



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import backend as K

from keras.layers import Layer

from keras.regularizers import L1L2



from imblearn import keras

from imblearn.keras import BalancedBatchGenerator

from imblearn.over_sampling import RandomOverSampler
def get_data():

    train_df = pd.read_csv('../input/train.csv')

    test_df = pd.read_csv('../input/test.csv')

    

    train_data = train_df.values

    test_data = test_df.values



    train_features = np.float64(train_data[:, 2:])

    test_features = np.float64(test_data[:, 1:])

    

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(np.concatenate([train_features, test_features], axis=0))

    train_features = scaler.transform(train_features)

    test_features = scaler.transform(test_features)

    

    train_target = np.float64(train_data[:, 1])

    

    test_ids = test_data[:, 0]

    

    return train_features, train_target, test_features, test_ids
train_features, train_target, test_features, test_ids = get_data()
def binary_crossentropy(y_true, y_pred):

    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
"""

Successfull architecture :



model = Sequential()

model.add(Dense(10, input_shape=(train_features.shape[1],), activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(15, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

"""



model = Sequential()

model.add(Dense(100, input_shape=(train_features.shape[1],), activation='selu')) # 40

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['acc'])
split_point = np.int32(0.8*len(train_features))

training_generator = BalancedBatchGenerator(train_features[:split_point], train_target[:split_point], batch_size=10000, random_state=42)

callback_history = model.fit_generator(generator=training_generator, validation_data=(train_features[split_point:], train_target[split_point:]), epochs=100)
predictions = model.predict(test_features).reshape((len(test_ids)))

submission = pd.DataFrame(np.transpose(np.array([test_ids, predictions])))

submission.columns = ['ID_code', 'target']

submission.to_csv('submission.csv', index=False)
submission.head()