import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn import metrics

from imblearn.over_sampling import RandomOverSampler

import os
import gc
#print(os.listdir("../input"))
import os
print(os.listdir("../input"))
def combine_data():
    train = pd.read_csv("../input/train.csv")
    train['isTrain'] = True
    train['isTest'] = False
    test = pd.read_csv("../input/test.csv")
    test['isTest'] = True
    test['isTrain'] = False
    df=train.append(test, ignore_index=True,sort=False)
    print(df.shape)
    return(df)
df = combine_data()
gc.collect()
df.head()
gc.collect()
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
X=scaler.fit_transform(df.loc[:,'48df886f9':].values)
input_data = Input(shape=(4993,))
encoded = Dense(256, activation='relu',name='encoded_layer')(input_data)
encoded = BatchNormalization()(encoded)
#encoded = Dense(32, activation='relu')(encoded)
#encoded = BatchNormalization()(encoded)
#encoded = Dense(16, activation='relu')(encoded)
#encoded = BatchNormalization(name='encoded_layer')(encoded)

#decoded = Dense(32, activation='relu')(encoded)
#decoded = BatchNormalization()(decoded)
#decoded = Dense(64, activation='relu')(decoded)
#decoded = BatchNormalization()(decoded)
decoded = Dense(4993, activation='linear')(encoded)
#decoded = Dense(82, activation='linear')(encoded)

autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.fit(X,X,
                epochs=10,
                batch_size=1024,
                shuffle=True)
#df.loc[:,'48df886f9':] = X 
#X = df.loc[df.isTrain,'48df886f9':].values
gc.collect()
scaler_y = MinMaxScaler(copy=False, feature_range=(0, 1))
y=scaler_y.fit_transform(df.loc[df.isTrain,'target'].values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X[df.isTrain], y, test_size=0.10, random_state=42)
model_1 = Sequential()
model_1.add(Dense(128, input_dim=256, activation='relu'))
model_1.add(BatchNormalization())
model_1.add(Dropout(0.2))
model_1.add(Dense(96, activation='relu'))
model_1.add(BatchNormalization())
model_1.add(Dropout(0.2))
model_1.add(Dense(64, activation='relu'))
model_1.add(BatchNormalization())
model_1.add(Dropout(0.2))
model_1.add(Dense(32, activation='relu'))
model_1.add(BatchNormalization())
model_1.add(Dropout(0.2))
model_1.add(Dense(16, activation='relu'))
model_1.add(BatchNormalization())
model_1.add(Dense(1, activation='linear'))

model_1.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])
model_1.summary()
intermediate_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_layer').output)

model_1.fit(intermediate_layer_model.predict(X_train), 
          y_train, 
          epochs=10,
          batch_size=32,
          validation_data=(intermediate_layer_model.predict(X_test),y_test),
          shuffle=True,
          verbose=1)
submission = pd.DataFrame()
submission['ID'] =  df.loc[df.isTest,'ID']
submission['target'] = scaler_y.inverse_transform(model_1.predict(intermediate_layer_model.predict(X[df.isTest])))
submission.to_csv('submission_1.csv', index=False)