import pandas as pd

import numpy as np
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')



print(f'TRAIN X: {train_features.shape}')

print(f'TEST X: {test_features.shape}')

print(f'TRAIN Y: {train_target.shape}')
train_features.head()
train_features.select_dtypes('object').info()
train_features.select_dtypes('object').describe()
train_target.head()
len(list(train_target.filter(regex=r'_inhibitor$').columns))
from sklearn.model_selection import train_test_split

import tensorflow as tf
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(875,))

model.add(tf.keras.layers.Dense(5120, activation='relu'))

model.add(tf.keras.layers.Dropout(0.6))

model.add(tf.keras.layers.Dense(1024, activation='relu'))

model.add(tf.keras.layers.Dropout(0.7))

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.7))

model.add(tf.keras.layers.Dense(206, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam')
train_x = train_features.drop(columns=['sig_id'])

train_x['cp_type'] = train_x['cp_type'].replace({'trt_cp': 0, 'ctl_vehicle':1})

train_x['cp_dose'] = train_x['cp_dose'].replace({'D1': 0, 'D2': 1})



test_x = test_features.drop(columns=['sig_id'])

test_x['cp_type'] = test_x['cp_type'].replace({'trt_cp': 0, 'ctl_vehicle':1})

test_x['cp_dose'] = test_x['cp_dose'].replace({'D1': 0, 'D2': 1})



train_y = train_target.drop(columns=['sig_id'])
train_x
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
num_epochs = 50

# history_lstm = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

history_lstm = model.fit(train_x, train_y, epochs=num_epochs)
op = pd.DataFrame((model.predict(test_x)), columns=train_target.drop(columns=['sig_id']).columns)

op['sig_id'] = test_features['sig_id']



column_order = ['sig_id']+list(train_target.drop(columns=['sig_id']).columns)
column_order
op
op[column_order].to_csv('submission.csv', index=False)