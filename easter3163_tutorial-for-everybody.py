# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/tobigs13nn/sample_submission.csv")

test = pd.read_csv("../input/tobigs13nn/test_df.csv")

train = pd.read_csv("../input/tobigs13nn/train_df.csv")



print(f"Train data shape {train.shape}")

print(f"Test data shape {test.shape}")
train.head()
test.head()
X = train.iloc[:,1:].values

y = train.iloc[:,0].values
import tensorflow as tf

y = tf.keras.utils.to_categorical(y)

y.shape
X = X / 255

X
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X, y, train_size=0.95,random_state=10)
import matplotlib.pyplot as plt

print(x_train.shape, y_train[0].argmax())

print(y_train[0])

plt.imshow(x_train[0].reshape(28, 28, 1)[:,:,0])
num_calsses = 10

model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(256,input_shape=(784,)),

  tf.keras.layers.BatchNormalization(),     

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(128,activation="selu",bias_initializer=tf.keras.initializers.he_normal(seed=None)), 

  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(64,activation="selu",bias_initializer=tf.keras.initializers.he_normal(seed=None)),

  tf.keras.layers.Dense(num_calsses,activation="softmax") 

])



model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

model.summary()
history = model.fit(x_train,y_train,

                    batch_size=512,

                    epochs=5,

                    validation_data=[x_val,y_val])
x_test = test.iloc[:,1:].values
x_test = x_test / 255
predictions = model.predict_classes(x_test)
sample_submission['Category'] = pd.Series(predictions)

sample_submission.head()
sample_submission.to_csv("submission.csv",index=False)
def cv_ensemble(X, y, x_test):

    preds = []

    from sklearn.model_selection import KFold, StratifiedKFold

    folds = KFold(n_splits=10, shuffle=True, random_state=2020)

    for train_idx, val_idx in folds.split(X):

        x_train, x_val = X[train_idx], X[val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        num_calsses = 10

        model = tf.keras.models.Sequential([

                  tf.keras.layers.Dense(256,input_shape=(784,)),

                  tf.keras.layers.BatchNormalization(),     

                  tf.keras.layers.Dropout(0.2),

                  tf.keras.layers.Dense(128,activation="selu",bias_initializer=tf.keras.initializers.he_normal(seed=None)), 

                  tf.keras.layers.BatchNormalization(),

                  tf.keras.layers.Dropout(0.2),

                  tf.keras.layers.Dense(64,activation="selu",bias_initializer=tf.keras.initializers.he_normal(seed=None)),

                  tf.keras.layers.Dense(num_calsses,activation="softmax") 

         ])



        model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

        model.fit(x_train,y_train,

                    batch_size=512,

                    epochs=30,

                    validation_data=[x_val,y_val])

        predictions = model.predict(x_test)

        preds.append(predictions)

    return preds
preds = cv_ensemble(X, y, x_test)
preds = np.array(preds)

print(preds.shape)
pred = np.mean(preds,axis=0)

print(pred.shape)
pred = np.argmax(pred, axis=1)

print(pred.shape)
sample_submission['Category'] = pd.Series(predictions)

sample_submission.head()
sample_submission.to_csv("submission_cv.csv",index=False)