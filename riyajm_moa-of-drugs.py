import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation,Dense,Dropout,BatchNormalization,Input
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

X =pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
y =pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
sample_submission=pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
from sklearn.preprocessing import LabelEncoder
X = X.iloc[: , 1: ].values
y = y.iloc[:,1:].values
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
print(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15)
model = Sequential([
    
    Input(X_train.shape[1]),
    layers.BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    #Dense(206, activation="relu"),
    #layers.BatchNormalization(),
    #layers.Dropout(0.5),
    Dense(y_train.shape[1], activation ="sigmoid")
    
    
])
model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()
def func(arg):
    
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg
model.fit(func(X_train),func(y_train) , verbose=2, epochs=1000, 
         validation_data=(func(X_train),func(y_train)),batch_size=8192)
X_final = test.iloc[: , 1: ].values
le = LabelEncoder()
X_final[:,2] = le.fit_transform(X_final[:,2])
le = LabelEncoder()
X_final[:,0] = le.fit_transform(X_final[:,0])
print(X_final)
y_pred = model.predict(func(X_final))

columns = list(sample_submission.columns)
columns.remove('sig_id')

for i in range(len(columns)):
    sample_submission[columns[i]] = y_pred[:, i]

sample_submission.to_csv('submission.csv', index=False)