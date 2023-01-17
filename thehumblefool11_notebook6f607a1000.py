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
data_df = pd.read_csv("../input/lish-moa/train_features.csv")

data_df1 = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

data_df2 = pd.read_csv("../input/lish-moa/test_features.csv")
data_df2.head()
data_df2["cp_type"].value_counts()
data_df.head()
data_df["cp_type"].value_counts()

c = data_df
data_df.value_counts()
data_df1.head()
data_df1.value_counts()
import tensorflow as tf

from tensorflow import keras
X_train = data_df.iloc[:, 1:]

y_train = data_df1.iloc[:,1:]
X_train
y_train
X_train.corr()
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()

a = X_train[["cp_dose"]]

a_enc = ord_enc.fit_transform(a)

a_enc
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()

c1 = c[["cp_type"]]

c_enc = ord_enc.fit_transform(c1)

c_enc
X_train["c_enc1"] = c_enc
X_train.drop(['cp_dose'], inplace = True, axis = 1)
X_train["a_enc1"] = a_enc
X_train
X_train["cp_time"].value_counts()
from sklearn.preprocessing import OneHotEncoder

or_enc1 = OneHotEncoder()

b = X_train[["cp_time"]]

b1=or_enc1.fit_transform(b)

b1
b1.toarray()[:,1]
X_train.drop(["cp_time"], inplace = True, axis = 1)
X_train.drop(["cp_type"], inplace = True, axis = 1)
X_train["24"] = b1.toarray()[:, 0]

X_train["48"] = b1.toarray()[:, 1]

X_train["72"] = b1.toarray()[:, 2]
X_train
y_train
X_train.shape
y_train.shape
from sklearn.model_selection import train_test_split

X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
print(X_train1.shape)

print(y_train1.shape)

print(X_test.shape)

print(y_test.shape)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau

import tensorflow_addons as tfa

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss
def create_model(num_columns):

    model = tf.keras.Sequential([

    tf.keras.layers.Input(num_columns),

    tf.keras.layers.BatchNormalization(),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))

    ])

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),

                  loss='binary_crossentropy', 

                  )

    return model

def metric(y_true, y_pred):

    metrics = []

    for _target in train_targets.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target], labels=[0,1]))

    return np.mean(metrics)
N_STARTS = 9

tf.random.set_seed(2020)



#res = y_train1.copy()

#submission.loc[:, train_targets.columns] = 0

#res.loc[:, train_targets.columns] = 0



for seed in range(N_STARTS):

    for n, (tr, te) in enumerate(KFold(n_splits=5, random_state=seed, shuffle=True).split(y_train1)):

        print(f'Fold {n}')

    

        model = create_model(len(X_train1.columns))

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-5, mode='min')

        model.fit(X_train1,

                  y_train1,

                  validation_data=(X_test, y_test),

                  epochs=150, batch_size=64,

                  callbacks=[reduce_lr_loss], verbose=2

                 )
data_df2
data_df2.drop(["sig_id"], inplace = True, axis = 1)
from sklearn.preprocessing import OrdinalEncoder

ord_enc11 = OrdinalEncoder()

a11 = data_df2[["cp_dose"]]

a_enc11 = ord_enc11.fit_transform(a11)

a_enc11
from sklearn.preprocessing import OrdinalEncoder

ord_enc12 = OrdinalEncoder()

c11 = data_df2[["cp_type"]]

c_enc11 = ord_enc12.fit_transform(c11)

c_enc11
data_df2.drop(["cp_type"], inplace = True, axis = 1)
data_df2["c_enc1"] = c_enc11
data_df2.drop(['cp_dose'], inplace = True, axis = 1)
data_df2["a_enc1"] = a_enc11
from sklearn.preprocessing import OneHotEncoder

or_enc11 = OneHotEncoder()

b11 = data_df2[["cp_time"]]

b12=or_enc11.fit_transform(b11)

b12
data_df2["24"] = b12.toarray()[:, 0]

data_df2["48"] = b12.toarray()[:, 1]

data_df2["72"] = b12.toarray()[:, 2]
data_df2.drop(["cp_time"], inplace = True, axis = 1)
data_df2.head()
X_pred= model.predict(data_df2)
X_pred1 = np.argmax(X_pred, axis =1)
X_pred2 = tf.keras.utils.to_categorical(X_pred1)
X_pred
data_df3 = pd.read_csv("../input/lish-moa/sample_submission.csv")
data_df3.head()
data_df3.iloc[:,1:] = X_pred

data_df3.to_csv('submission.csv', index=False)
#from sklearn.ensemble import RandomForestClassifier

#rfc = RandomForestClassifier()

#rfc.fit(X_train1, y_train1)