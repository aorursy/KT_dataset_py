import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Dropout

from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

from tensorflow.keras.callbacks import Callback

import tensorflow.keras.backend as K

from tensorflow.keras.layers import BatchNormalization

import math

from tensorflow.keras import optimizers

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/kaggle-club-stockpriceprediction/train.csv")

train.head()
sample = pd.read_csv("/kaggle/input/kaggle-club-stockpriceprediction/sample_submission.csv")

sample.head()
train.tail(10)
train_y = train[(train["date"] == "2020-08-07")|(train["date"] == "2020-07-31")].groupby("name")["end"].diff().dropna().map(lambda x: 1 if x >0 else 0)

test_y = train[(train["date"] == "2020-08-14")|(train["date"] == "2020-08-07")].groupby("name")["end"].diff().dropna().map(lambda x: 1 if x >0 else 0)
train_X = train[train["date"] < "2020-07-31"].iloc[:,1:]

test_X = train[(train["date"] < "2020-08-07")&(train["date"] > "2018-01-12")].iloc[:,1:]

sub_X = train[(train["date"] < "2020-08-14")&(train["date"] > "2018-01-17")].iloc[:,1:]
le = LabelEncoder()

le = le.fit(train_X['market'])

train_X['market'] = le.transform(train_X['market'])

test_X['market'] = le.transform(test_X['market'])

sub_X['market'] = le.transform(sub_X['market'])
train_X
using_col = ['name', 'open', 'high', 'low', 'end', 'volume', 'end_fixed','market','open_diff', 'high_diff', 'low_diff', 'end_diff', 'volume_diff', 'end_fixed_diff']

diff_col = ['open', 'high', 'low', 'end', 'volume', 'end_fixed']

diff_col_new = ['open_diff', 'high_diff', 'low_diff', 'end_diff', 'volume_diff', 'end_fixed_diff']
train_X_diff = train_X.groupby("name")[diff_col].diff()

train_X_diff.columns = diff_col_new

test_X_diff = test_X.groupby("name")[diff_col].diff()

test_X_diff.columns = diff_col_new

sub_X_diff = sub_X.groupby("name")[diff_col].diff()

sub_X_diff.columns = diff_col_new
train_X = pd.concat([train_X,train_X_diff],axis=1).dropna()

test_X = pd.concat([test_X,test_X_diff],axis=1).dropna()

sub_X = pd.concat([sub_X,sub_X_diff],axis=1).dropna()
train_X
len(train_X["date"].unique())
train_X = train_X[using_col].values.reshape([-1,len(train_X["date"].unique()),14])

test_X = test_X[using_col].values.reshape([-1,len(test_X["date"].unique()),14])

sub_X = sub_X[using_col].values.reshape([-1,len(sub_X["date"].unique()),14])
sv = ModelCheckpoint(

        'model.h5', monitor='val_loss', verbose=2, save_best_only=True,

        save_weights_only=True, mode='max', save_freq='epoch')



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=10, min_lr=0.00001)
model = Sequential()

model.add(Dense(128,input_shape=(train_X.shape[1:]),activation="relu"))

model.add(Dropout(0.25))

model.add(BatchNormalization(axis=-1))

model.add(Dense(128,input_shape=(train_X.shape[1:]),activation="relu"))

model.add(Dropout(0.25))

model.add(BatchNormalization(axis=-1))

model.add(Dense(128,input_shape=(train_X.shape[1:]),activation="relu"))

model.add(Dropout(0.25))

model.add(BatchNormalization(axis=-1))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy',metrics=["AUC"])

model.summary()
train_X = train_X.astype(np.float64)

train_y = train_y.astype(np.float64)

sub_X = sub_X.astype(np.float64)

test_X = test_X.astype(np.float64)

test_y = test_y.astype(np.float64)
history = model.fit(train_X, train_y,

                    epochs= 10,

                    validation_data=(test_X,test_y),

                    batch_size=128,callbacks=[sv,reduce_lr])
roc_auc_score(test_y,model.predict(test_X).reshape(-1))
#predict

sample["target"] = model.predict(sub_X).reshape(-1)

sample.to_csv("owari.csv",index=False)