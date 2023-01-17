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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



X = train.drop('label', axis=1)

y = train.label



X = X / 255  #normalization

test = test / 255   #normalization
X = X.values.reshape(-1, 784)

y = y.values.reshape(-1, 1)



X = X.astype('float64')

y = y.astype('float64')

test = test.astype('float64')
from tensorflow import keras

my_model = keras.Sequential([

    keras.layers.Dense(10, 'softmax')

])



my_model.compile(loss='sparse_categorical_crossentropy',

                 optimizer='adam',

                 metrics=['accuracy'])
my_model.fit(X, y,

             batch_size=20,

             epochs=20,

             verbose=0)
pred_test = my_model.predict(test)

pred_test = pred_test.argmax(axis=1)



submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission['Label'] = pred_test

submission.to_csv('sample_submission.csv', index=False)