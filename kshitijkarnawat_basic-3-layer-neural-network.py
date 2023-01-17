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
import tensorflow as tf

from tensorflow import keras
df = pd.read_csv('../input/digit-recognizer/train.csv')
df.head()
X_train = df.iloc[:,1:]

y_train = df.iloc[:,0]
test_df = pd.read_csv('../input/digit-recognizer/test.csv')

test_df.head()
X_train = X_train / 255

test_df = test_df / 255
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),

                          keras.layers.Dense(128, activation=tf.nn.relu),

                          keras.layers.Dense(10,activation=tf.nn.softmax)

                          ])
model.compile(optimizer = tf.optimizers.Adam(),

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=5)
pred = model.predict(test_df)

pred = np.argmax(pred, axis = 1)

print(pred[0]) #change the index value of pred to view different results