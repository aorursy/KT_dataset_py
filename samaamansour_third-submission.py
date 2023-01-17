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
train_set = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_set = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_set.head()
import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras.layers.core import Dense
model_lstm1 = tf.keras.Sequential([

                             tf.keras.layers.Embedding(5000, 20),

                             tf.keras.layers.LSTM(20),

                             tf.keras.layers.Dense(1, activation='sigmoid')

    

    

])

model_lstm1.summary()

model_lstm1.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
y = train_set['target']

features = ['keyword']

X = pd.get_dummies(train_set[features])

X_test = pd.get_dummies(test_set[features])

np.shape(X_test)

model_lstm1.fit(X , y , epochs=20)

predictions = model_lstm1.predict(X_test)

new = predictions.ravel()



print(np.shape(new))

print(new.ndim)

new[new > 0.5] = int(1)

new[new < 0.5] = int(0)

new = new.astype(int)

print(new)

print(new)

output = pd.DataFrame({'id': test_set.id, 'target': new})

output.to_csv('my_submission.csv', index=False)

print("Submission saved!")