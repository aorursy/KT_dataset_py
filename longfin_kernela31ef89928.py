# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!ls -al ../input/petfinder-adoption-prediction
train_csv = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")

train_csv['AdoptionSpeed'].hist()
train_csv.head()
features = ["Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee"]
mask =  np.random.rand(len(train_csv)) < 0.8

x = train_csv[mask].filter(items=features)

y = train_csv[mask].filter(items=["AdoptionSpeed"])

valid_x = train_csv[~mask].filter(items=features)

valid_y = train_csv[~mask].filter(items=["AdoptionSpeed"])
from keras import backend as K

import tensorflow as tf



def rmse(y, y_pred):

    return K.sqrt(K.mean(K.square(tf.cast(y, tf.float32)-y_pred), axis=-1))
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, CuDNNLSTM, Conv1D, Add

from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Lambda, Concatenate

from keras.optimizers import Nadam, Adam

from keras.models import Model, Sequential



model = Sequential()

model.add(Dense(30, input_dim=len(features)))

model.add(Dense(100, input_dim=30, activation='relu'))

model.add(Dense(1, activation='linear'))



model.compile(

    loss='mse',

    optimizer=Adam(lr=0.03)

)
model.fit(x, y, validation_data=(valid_x, valid_y), batch_size=256, epochs=100)
!ls -al ../input/petfinder-adoption-prediction/test
test_csv = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")

test_x = test_csv.filter(items=features)

pred_y = np.clip(model.predict(test_x), 0, 4).round().squeeze().astype(int)

pred_y
submission_df = pd.DataFrame(data={"PetID": test_csv["PetID"], "AdoptionSpeed": pred_y})

submission_df.to_csv("submission.csv", index=False)

submission_df["AdoptionSpeed"].hist()