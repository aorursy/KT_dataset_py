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
import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split
Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test_data = pd.read_csv("../input/Kannada-MNIST/test.csv")

train_data = pd.read_csv("../input/Kannada-MNIST/train.csv")
X = train_data.drop('label', axis=1)

Y = train_data['label']

X = X.values.reshape(-1, 28, 28)

Y = Y.values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

x_final_test = test_data.drop("id", axis=1)

x_final_test = x_final_test.values.reshape(-1, 28, 28)
model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)    
model.fit(x_train, y_train, epochs=20)



model.evaluate(x_test, y_test, verbose=2)
y_hat = model.predict(x_final_test)
y_hat.shape
y_hat = np.argmax(y_hat,axis=1)
y_hat.shape
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sample_sub['label'] = y_hat

sample_sub.to_csv('submission.csv',index=False)