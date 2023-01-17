import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
SEED = 42

EPOCHS = 300

BATCH_SIZE = 32



df = pd.read_csv('../input/parkinsons-disease-data-set/parkinsons.data')
def seed_everything(seed):

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



seed_everything(SEED)
df
df['status'].value_counts()
df.describe()
df = df.drop('name', axis=1)



train_df, test_df = train_test_split(df, 

                                     test_size=0.2, 

                                     random_state=SEED)

train_df, val_df = train_test_split(train_df,

                                    test_size=0.2,

                                    random_state=SEED)





X_train = train_df.drop('status', axis=1).values.astype('float32')

y_train = train_df['status'].values.astype('int32')

X_val = val_df.drop('status', axis=1).values.astype('float32')

y_val = val_df['status'].values.astype('int32')

X_test = test_df.drop('status', axis=1).values.astype('float32')

y_test = test_df['status'].values.astype('int32')



mmsc = MinMaxScaler()

X_train = mmsc.fit_transform(X_train) 

X_val = mmsc.transform(X_val)

X_test = mmsc.transform(X_test)



y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

y_test = to_categorical(y_test)
shape = X_train.shape[1]

num_classes = y_train.shape[1]



model = Sequential()

model.add(Input((shape,)))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

opt = tfa.optimizers.RectifiedAdam()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()
es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 

                    validation_data=(X_val, y_val), callbacks=[es_callback])
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()

plt.show()
model.evaluate(X_test, y_test)