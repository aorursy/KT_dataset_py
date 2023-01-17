# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
features = train.columns.values[2:202]

sns.distplot(train[features].mean(axis=0),color="blue", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical



def create_model():



    model = Sequential()

    model.add(Dense(units=5, activation='relu', input_dim=200))

    model.add(Dense(units=1, activation='sigmoid'))



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
x_train = train[features].values

y_train = train.target.values
# Implememt stratified k-Fold to preform cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(x_train, y_train)
loss = []

val_loss = []



def plot_history(loss, val_loss):

    plt.plot(loss)

    plt.plot(val_loss)

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
# for train_index, test_index in skf.split(x_train, y_train):

#     print("TRAIN:", train_index, "TEST:", test_index)

#     x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]

#     y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

#     model = None

#     model = create_model()

#     history = model.fit(x_train_cv, y_train_cv, epochs=300, batch_size=2500, validation_data=(x_test_cv, y_test_cv))

#     plot_history(history.history['loss'], history.history['val_loss'])

#     break
model = create_model()

model.fit(x_train, y_train, epochs=10, batch_size=2500)
x_test = test[features].values

classes = model.predict(x_test)

predictions = np.where(classes > 0.5, 1, 0)
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission_3.csv", index=False)