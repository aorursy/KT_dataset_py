# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")

test = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")
train.head()
len(train)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics: 

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max< np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
timestep = 1000

X = train.signal.values.reshape(-1, timestep, 1)

y = train.open_channels.values.reshape(-1, timestep, 1)
np.random.seed(42)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
X_valid.shape
y_valid.shape
X_test = test.signal.values.reshape(-1, timestep, 1)
X_train.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers  import LSTM

from tensorflow.keras.layers  import Dense

from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras import callbacks

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import GRU

from tensorflow.keras import Model

from tensorflow.keras.layers import RepeatVector

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Conv1D
def build_model(n_classes, seq_len=500, n_units=256):

    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]))

    model.add(Bidirectional(GRU(n_units,return_sequences=True)))

    Dropout(0.3)

    model.add(Bidirectional(GRU(n_units,return_sequences=True)))

    Dropout(0.3)

    model.add(Dense(n_classes,activation="softmax"))          

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
n_classes = train.open_channels.unique().shape[0]

model = build_model(n_classes, timestep)

model.summary()
model.summary()
model.fit(X_train,y_train,batch_size=64,epochs=60, callbacks=[callbacks.ReduceLROnPlateau(),callbacks.ModelCheckpoint('model.h5')],validation_data=(X_valid, y_valid))
from sklearn.metrics import f1_score

model.load_weights('model.h5')

valid_pred = model.predict(X_valid, batch_size=64).argmax(axis=-1)

f1_score(y_valid.reshape(-1), valid_pred.reshape(-1), average='macro')
sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype=dict(time=str))
test_pred = model.predict(X_test, batch_size=64).argmax(axis=-1)

sub.open_channels = test_pred.reshape(-1)

sub.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
X_train = X_train.reshape(-1,1)
clf = RandomForestClassifier()

clf.fit(X_train,y_train)