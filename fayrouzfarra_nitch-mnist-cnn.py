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
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, models

tf.__version__
train_data = pd.read_csv('../input/nicht-mnist/train.csv',header=None, index_col =0)

test_data = pd.read_csv('../input/nicht-mnist/test.csv',header=None , index_col = 0)
train_data
train_data[1].unique()
len(train_data[1].unique())
X_train = train_data.iloc[:,1:]   

X_train
y_train = train_data.iloc[:,0]    ##y_train = train_data[1]

y_train.head()
len(y_train.unique())
X_train = X_train / 255.0

test_data = test_data / 255.0
y_train.value_counts()
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

y_train = pd.DataFrame(label_encoder.fit_transform(y_train))

y_train
from sklearn.model_selection import train_test_split

X_train, X_val ,y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=1)
X_train_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in X_train.iterrows()] ] )

X_val_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in X_val.iterrows()] ] )
X_train.shape , X_train_np.shape
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

model = tf.keras.models.Sequential()



model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train_np, y_train, epochs=20, 

                    validation_data=(X_val_np, y_val), callbacks=[earlystop])
model.evaluate(X_val_np,  y_val, verbose=2)
X_test_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in test_data.iterrows()] ] )

preds = np.argmax(model.predict(X_test_np), axis=1).tolist()

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

pred_labes = pd.Series([class_labels[p] for p in preds])

pred_labes
out_df = pd.DataFrame({

    'Id': test_data.index,

    'target': pred_labes

})

out_df
out_df.to_csv('my_submission.csv', index=False)