# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_targets_scored.head()
train_targets_scored.sum()[1 : ].sort_values()
from sklearn.preprocessing import LabelEncoder

X = train_features.iloc[: , 1: ].values

y = train_targets_scored.iloc[:,1:].values

# X = 23814 * 874   Y = 23814 * 206



le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])

le = LabelEncoder()

X[:,0] = le.fit_transform(X[:,0])

print(X)

print(X.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Activation,Dense,Dropout,BatchNormalization,Input

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras import regularizers
model = Sequential([

    Input(X_train.shape[1]),

    layers.BatchNormalization(),

    Dropout(0.2),

    Dense(1024,bias_regularizer=regularizers.l2(1e-4), activation="relu"),

    layers.BatchNormalization(),

    layers.Dropout(0.3),

    Dense(1024,bias_regularizer=regularizers.l2(1e-4), activation="relu"),

    layers.BatchNormalization(),

    layers.Dropout(0.3),

    Dense(y_train.shape[1],activity_regularizer=regularizers.l2(1e-5), activation ="sigmoid"),

    

    

])

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)
model.summary()
def my_func(arg):

  arg = tf.convert_to_tensor(arg, dtype=tf.float32)

  return arg
model.fit(my_func(X_train),my_func(y_train) , verbose=2, epochs=50, 

         validation_data=(my_func(X_train),my_func(y_train)),batch_size=32,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor='val_loss', 

            factor=0.3, 

            patience=3,

            epsilon = 1e-4, 

            mode = 'min',

            verbose=1

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            min_delta=0,

            patience=10,

            mode='auto',

            verbose=1,

            baseline=None,

            restore_best_weights=True

        )

    ]

         )
X_final = test_features.iloc[: , 1: ].values

le = LabelEncoder()

X_final[:,2] = le.fit_transform(X_final[:,2])

le = LabelEncoder()

X_final[:,0] = le.fit_transform(X_final[:,0])

print(X_final)
y_pred = model.predict(my_func(X_final))



columns = list(sample_submission.columns)

columns.remove('sig_id')



for i in range(len(columns)):

    sample_submission[columns[i]] = y_pred[:, i]



sample_submission.to_csv('submission.csv', index=False)