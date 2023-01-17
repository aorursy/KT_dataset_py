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
from sklearn.preprocessing import MinMaxScaler, Imputer
data = pd.read_csv('../input/train.csv')

x = data.drop(['ID_code','target'], axis = 1)

y = data['target']

features = x.columns



scaler = MinMaxScaler()

x = scaler.fit_transform(x)



x = pd.DataFrame(x)

x = x[(x>0.05)&(x<0.95)]



imputer = Imputer()

x = imputer.fit_transform(x)



x = pd.DataFrame(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)



from sklearn.preprocessing import StandardScaler

x_train = StandardScaler().fit_transform(x_train)

x_test = StandardScaler().fit_transform(x_test)



from scipy.ndimage.filters import uniform_filter1d

x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=1)], axis=2)

x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=1)], axis=2)
from keras.models import Model

from keras.layers import CuDNNLSTM, CuDNNGRU, Dropout, Dense, GlobalMaxPool1D, Input, Bidirectional

from keras.layers.merge import concatenate

from keras.callbacks import ModelCheckpoint
inp = Input(shape = x_train.shape[1:])

x = Bidirectional(CuDNNLSTM(units = 256, return_sequences = True))(inp)

x = Dropout(0.5)(x)

x = Bidirectional(CuDNNLSTM(units = 128, return_sequences = True))(x)

x = Dropout(0.5)(x)

x = Bidirectional(CuDNNLSTM(units = 64, return_sequences = True))(x)

x = Dropout(0.5)(x)

x = Bidirectional(CuDNNLSTM(units = 32, return_sequences = True))(x)

y = Bidirectional(CuDNNLSTM(units = 256, return_sequences = True))(inp)

y = Dropout(0.5)(y)

y = Bidirectional(CuDNNLSTM(units = 128, return_sequences = True))(y)

y = Dropout(0.5)(y)

y = Bidirectional(CuDNNLSTM(units = 64, return_sequences = True))(y)

y = Dropout(0.5)(y)

y = Bidirectional(CuDNNLSTM(units = 32, return_sequences = True))(y)

x = concatenate([x,y])

x = GlobalMaxPool1D()(x)

x = Dropout(0.5)(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.25)(x)

x = Dense(64, activation='relu')(x)

x = Dense(1, activation='sigmoid')(x)



model = Model(inp,x)
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"

checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')

callbacks=[checkpoint]



model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,

                           validation_data = (x_test,y_test), 

                           epochs = 20,

                           batch_size = 256,callbacks = callbacks)
test = pd.read_csv("../input/test.csv")

sample_sub = pd.read_csv("../input/sample_submission.csv")

test = MinMaxScaler().fit_transform(test.drop('ID_code',axis = 1))

test = StandardScaler().fit_transform(test)

test = np.stack([test, uniform_filter1d(test, axis=1, size=1)], axis=2)

sample_sub["target"] = model.predict(test)
sample_sub.to_csv('../input/8thsubmission.csv', encoding='utf-8', index=False)