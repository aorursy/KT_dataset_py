import numpy as np

import pandas as pd



N_FEATURES = 22

# taken from http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/

THRESHOLD = 73 
train_df = pd.read_csv("/kaggle/input/train.zip")

# to reduce memory consumption

train_df[train_df.columns[1:]] = train_df[train_df.columns[1:]].astype(np.float32)

train_df.shape
good_ids = set(train_df.loc[train_df['Ref'].notna(), 'Id'])

train_df = train_df[train_df['Id'].isin(good_ids)]

train_df.shape
train_df.reset_index(drop=True, inplace=True)

train_df.fillna(0.0, inplace=True)

train_df.head()
train_df.shape
train_df = train_df[train_df['Expected'] < THRESHOLD]

train_df.shape
train_groups = train_df.groupby("Id")

train_size = len(train_groups)
MAX_SEQ_LEN = train_groups.size().max()

MAX_SEQ_LEN
X_train = np.zeros((train_size, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)

y_train = np.zeros(train_size, dtype=np.float32)



i = 0

for _, group in train_groups:

    X = group.values

    seq_len = X.shape[0]

    X_train[i,:seq_len,:] = X[:,1:23]

    y_train[i] = X[0,23]

    i += 1

    del X

    

del train_groups

X_train.shape, y_train.shape
test_df = pd.read_csv("/kaggle/input/test.zip")

test_df[test_df.columns[1:]] = test_df[test_df.columns[1:]].astype(np.float32)

test_ids = test_df['Id'].unique()



# Convert all NaNs to zero

test_df = test_df.reset_index(drop=True)

test_df = test_df.fillna(0.0)
test_groups = test_df.groupby("Id")

test_size = len(test_groups)



X_test = np.zeros((test_size, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)



i = 0

for _, group in test_groups:

    X = group.values

    seq_len = X.shape[0]

    X_test[i,:seq_len,:] = X[:,1:23]

    i += 1

    del X

    

del test_groups

X_test.shape
from keras.layers import (

    Input,

    Dense,

    LSTM,

    GlobalAveragePooling1D,

    AveragePooling1D,

    TimeDistributed,

    Flatten,

    Bidirectional,

    Dropout,

    Masking,

    Layer,

    BatchNormalization

)

from keras.models import Model

from keras.optimizers import Adam,Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=0.01)
BATCH_SIZE = 1024

N_EPOCHS = 30
class NonMasking(Layer):   

    def __init__(self, **kwargs):   

        self.supports_masking = True  

        super(NonMasking, self).__init__(**kwargs)   

    def build(self, input_shape):   

        input_shape = input_shape   

    def compute_mask(self, input, input_mask=None):   

        # do not pass the mask to the next layers   

        return None   

    def call(self, x, mask=None):   

        return x   

    def get_output_shape_for(self, input_shape):   

        return input_shape 
def get_model_deep(shape=(MAX_SEQ_LEN, N_FEATURES)):

    inp = Input(shape)

    x = Dense(16)(inp)

    x = BatchNormalization()(x)

    x = Masking(mask_value=0.)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = BatchNormalization()(x)

    x = TimeDistributed(Dense(64))(x)

    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    x = BatchNormalization()(x)

    x = TimeDistributed(Dense(1))(x)

    x = BatchNormalization()(x)

    x = NonMasking()(x)

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(1)(x)



    model = Model(inp, x)

    return model
model = get_model_deep()

model.compile(optimizer=Nadam(lr=0.001), loss='mae')

model.summary()
history = model.fit(X_train, y_train, 

            batch_size=BATCH_SIZE, epochs=N_EPOCHS, 

            validation_split=0.2, callbacks=[early_stopping, reduce_lr])
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



del history
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

submission = pd.DataFrame({'Id': test_ids, 'Expected': y_pred.reshape(-1)})

submission.to_csv('submission.csv', index=False)