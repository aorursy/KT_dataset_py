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
train_test_data_merged = pd.read_csv('../input/train_test_data_merged.csv')
train_df = train_test_data_merged[train_test_data_merged.type == "train"].drop("type", axis=1)

test_df = train_test_data_merged[train_test_data_merged.type == "test"].drop(["type", "SalePrice"], axis=1)
from sklearn.model_selection import train_test_split

from math import sqrt,log

X = train_df.drop('SalePrice', axis=1)

y = train_df['SalePrice'].apply(log)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
import numpy

import pandas

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from keras import regularizers

from keras.layers.normalization import BatchNormalization
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def model1():

    # create model

    model = Sequential()

    model.add(Dense(13, input_dim=251, kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01)))

    #model.add(BatchNormalization())

    model.add(Activation('relu'))

    #model.add(Dropout(0.2))

    model.add(Dense(5, kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01)))

    #model.add(BatchNormalization())

    model.add(Activation('relu'))

    #model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='glorot_normal'))

    # Compile model

    model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mae'])

    return model
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(13, input_dim=251, kernel_initializer='glorot_normal'))

    #model.add(BatchNormalization())

    model.add(Activation('relu'))

    #model.add(Dropout(0.5))

    model.add(Dense(1, kernel_initializer='glorot_normal'))

    # Compile model

    model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['mae'])

    return model
model = baseline_model()

history = model.fit(X_train, y_train, epochs=120, batch_size=256,  verbose=1, validation_split=0.1)
model = model1()

history = model.fit(X_train, y_train, epochs=120, batch_size=256,  verbose=1, validation_split=0.1)
import matplotlib.pyplot as plt

print(history.history.keys())

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
ynew = model.predict(test_df)

#print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
id_val = list(range(1461,2920))

op = pd.concat([pd.DataFrame(id_val),pd.DataFrame(ynew)], axis = 1)

op.columns = ['Id', 'SalePrice_log']

op['SalePrice'] = np.exp(op['SalePrice_log'])

op.drop('SalePrice_log', axis = 1, inplace = True)

op.head()

op.to_csv('submission.csv', index=False)
op.head()